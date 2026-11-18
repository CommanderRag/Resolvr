"""API tests against a real Postgres and a real Redis (no mocks for either).

    docker compose up -d db redis
    cd service
    TEST_DATABASE_URL=postgresql://resolvr:resolvr@localhost:5432/resolvr_test pytest -q

The database name must end in _test (it is created and migrated automatically): the tests
TRUNCATE tables, so they refuse to run against anything else. Redis tests use database 15
(override with TEST_REDIS_URL) and are skipped if Redis is not reachable.
The LLM/embedding engine is replaced by a fake so tests need no API key or model download.
"""
import asyncio
import os
import pathlib
import threading
import time
import uuid
from urllib.parse import urlparse, urlunparse

import asyncpg
import httpx
import pytest
import pytest_asyncio

from app import db
from app.cache import Cache
from app.engine import Result
from app.main import app

DSN = os.environ.get("TEST_DATABASE_URL")
REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEST_DATABASE_URL to run API tests")

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"


async def ensure_test_db(dsn: str) -> None:
    """Create the *_test database and apply the schema if they don't exist yet."""
    url = urlparse(dsn)
    name = url.path.lstrip("/")
    if not name.endswith("_test"):
        raise RuntimeError(f"refusing to run tests on database {name!r}: its name must end in _test")
    admin = await asyncpg.connect(urlunparse(url._replace(path="/postgres")))
    try:
        if not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    conn = await asyncpg.connect(dsn)
    try:
        if await conn.fetchval("SELECT to_regclass('public.tickets')") is None:
            await conn.execute(SCHEMA.read_text())
        await db.apply_upgrades(conn)
    finally:
        await conn.close()


class FakeEngine:
    def __init__(self):
        self.fail = False
        self.delay = 0.0
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def process(self, text: str) -> Result:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("llm unavailable")
        finally:
            with self._lock:
                self.active -= 1
        return Result("Delivery delay", False, "needs a human", f"echo: {text}", [(2, 0.9), (0, 0.5), (1, 0.1)])


@pytest_asyncio.fixture
async def client():
    await ensure_test_db(DSN)
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=10)
    await pool.execute("TRUNCATE ticket_grounding, tickets, grounding_pairs CASCADE")
    await pool.executemany(
        "INSERT INTO grounding_pairs (id, issue, resolution) VALUES ($1, $2, $3)",
        [(i, f"issue {i}", f"resolution {i}") for i in range(3)],
    )
    cache = Cache(REDIS_URL, ttl=60)
    await cache.clear()
    app.state.pool = pool
    app.state.engine = FakeEngine()
    app.state.cache = cache
    app.state.llm_slots = asyncio.Semaphore(2)   # cap simultaneous engine calls
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await cache.close()
    await pool.close()


@pytest_asyncio.fixture
async def needs_redis(client):
    if not await app.state.cache.ping():
        pytest.skip("Redis not reachable (docker compose up -d redis)")


async def wait_done(client, ticket_id, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        body = (await client.get(f"/tickets/{ticket_id}")).json()
        if body["status"] in ("done", "failed"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError("ticket did not finish in time")


async def submit(client, text):
    r = await client.post("/tickets", json={"text": text})
    assert r.status_code == 202
    return await wait_done(client, r.json()["id"])


# ---------- core behaviour -------------------------------------------------
async def test_create_then_fetch_result(client):
    r = await client.post("/tickets", json={"text": "my package is late"})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "received"
    ticket = await wait_done(client, body["id"])
    assert ticket["status"] == "done"
    assert ticket["intent"] == "Delivery delay"
    assert ticket["auto_handle"] is False
    assert ticket["reply"] == "echo: my package is late"
    assert ticket["latency_ms"] >= 0
    assert ticket["cache_hit"] is False


async def test_grounding_returned_in_rank_order(client):
    ticket = await submit(client, "where is my order")
    assert [g["rank"] for g in ticket["grounding"]] == [1, 2, 3]
    assert [g["issue"] for g in ticket["grounding"]] == ["issue 2", "issue 0", "issue 1"]


async def test_engine_failure_is_recorded_not_lost(client):
    app.state.engine.fail = True
    ticket = await submit(client, "help")
    assert ticket["status"] == "failed"
    assert "llm unavailable" in ticket["error"]
    assert ticket["reply"] is None


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 2001}])
async def test_invalid_text_rejected(client, payload):
    assert (await client.post("/tickets", json=payload)).status_code == 422


async def test_unknown_ticket_is_404(client):
    assert (await client.get(f"/tickets/{uuid.uuid4()}")).status_code == 404


async def test_malformed_id_is_422(client):
    assert (await client.get("/tickets/not-a-uuid")).status_code == 422


async def test_claim_is_atomic(client):
    """Two workers racing for one ticket: exactly one gets it."""
    row = await db.insert_ticket(app.state.pool, "hello", "h" * 64)
    claims = await asyncio.gather(*[db.claim_ticket(app.state.pool, row["id"]) for _ in range(10)])
    assert sum(c is not None for c in claims) == 1
    assert next(c for c in claims if c is not None)["text"] == "hello"


async def test_healthz(client):
    body = (await client.get("/healthz")).json()
    assert body["status"] == "ok"
    assert body["redis"] in ("up", "down")


# ---------- caching --------------------------------------------------------
async def test_repeat_message_served_from_cache(client, needs_redis):
    first = await submit(client, "my package is late")
    second = await submit(client, "my package is late")
    assert app.state.engine.calls == 1                 # second request never reached the LLM
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert second["reply"] == first["reply"] and second["intent"] == first["intent"]
    assert [g["issue"] for g in second["grounding"]] == [g["issue"] for g in first["grounding"]]


async def test_cache_key_ignores_case_and_surrounding_whitespace(client, needs_redis):
    await submit(client, "Where is my ORDER")
    again = await submit(client, "  where is my order ")
    assert again["cache_hit"] is True
    assert app.state.engine.calls == 1


async def test_failures_are_not_cached(client, needs_redis):
    app.state.engine.fail = True
    assert (await submit(client, "flaky message"))["status"] == "failed"
    app.state.engine.fail = False
    retry = await submit(client, "flaky message")
    assert retry["status"] == "done" and retry["cache_hit"] is False
    assert app.state.engine.calls == 2


async def test_concurrent_identical_messages_cost_one_llm_call(client, needs_redis):
    app.state.engine.delay = 0.4
    responses = await asyncio.gather(*[client.post("/tickets", json={"text": "burst message"}) for _ in range(6)])
    tickets = [await wait_done(client, r.json()["id"]) for r in responses]
    assert all(t["status"] == "done" for t in tickets)
    assert app.state.engine.calls == 1                 # single-flight: five requests waited for the first
    assert sum(t["cache_hit"] for t in tickets) == 5


async def test_redis_down_degrades_to_slower_not_broken(client):
    old = app.state.cache
    app.state.cache = Cache("redis://127.0.0.1:1/0")   # nothing listens on port 1
    try:
        ticket = await submit(client, "redis is down")
        assert ticket["status"] == "done" and ticket["cache_hit"] is False
        assert (await client.get("/healthz")).json()["redis"] == "down"
    finally:
        await app.state.cache.close()
        app.state.cache = old


async def test_stats_report_cache_effectiveness(client, needs_redis):
    await submit(client, "stats message")
    await submit(client, "stats message")
    await submit(client, "another one")
    s = (await client.get("/stats")).json()
    assert s["total"] == 3 and s["done"] == 3 and s["failed"] == 0
    assert s["cache_hits"] == 1 and s["cache_hit_rate"] == 0.333


async def test_llm_concurrency_is_capped(client):
    """A burst of different messages queues behind the cap instead of hammering the provider."""
    app.state.engine.delay = 0.2
    responses = await asyncio.gather(*[client.post("/tickets", json={"text": f"unique message {i}"}) for i in range(6)])
    tickets = [await wait_done(client, r.json()["id"]) for r in responses]
    assert all(t["status"] == "done" for t in tickets)
    assert app.state.engine.calls == 6
    assert app.state.engine.max_active <= 2
