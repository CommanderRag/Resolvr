"""API tests against a real Postgres (no mocks for the database).

    docker compose up -d db
    cd service && TEST_DATABASE_URL=postgresql://resolvr:resolvr@localhost:5432/resolvr_test pytest -q

The database name must end in _test (it is created and migrated automatically): the tests
TRUNCATE tables, so they refuse to run against anything else.

The LLM/embedding engine is replaced by a fake so tests need no API key or model download.
"""
import asyncio
import os
import pathlib
import uuid
from urllib.parse import urlparse, urlunparse

import asyncpg
import httpx
import pytest
import pytest_asyncio

from app import db
from app.engine import Result
from app.main import app

DSN = os.environ.get("TEST_DATABASE_URL")
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
    finally:
        await conn.close()


class FakeEngine:
    def __init__(self):
        self.fail = False

    def process(self, text: str) -> Result:
        if self.fail:
            raise RuntimeError("llm unavailable")
        return Result("Delivery delay", False, "needs a human", f"echo: {text}", [(2, 0.9), (0, 0.5), (1, 0.1)])


@pytest_asyncio.fixture
async def client():
    await ensure_test_db(DSN)
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=5)
    await pool.execute("TRUNCATE ticket_grounding, tickets, grounding_pairs CASCADE")
    await pool.executemany(
        "INSERT INTO grounding_pairs (id, issue, resolution) VALUES ($1, $2, $3)",
        [(i, f"issue {i}", f"resolution {i}") for i in range(3)],
    )
    app.state.pool = pool
    app.state.engine = FakeEngine()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await pool.close()


async def wait_done(client, ticket_id, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        body = (await client.get(f"/tickets/{ticket_id}")).json()
        if body["status"] in ("done", "failed"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError("ticket did not finish in time")


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


async def test_grounding_returned_in_rank_order(client):
    tid = (await client.post("/tickets", json={"text": "where is my order"})).json()["id"]
    ticket = await wait_done(client, tid)
    assert [g["rank"] for g in ticket["grounding"]] == [1, 2, 3]
    assert [g["issue"] for g in ticket["grounding"]] == ["issue 2", "issue 0", "issue 1"]


async def test_engine_failure_is_recorded_not_lost(client):
    app.state.engine.fail = True
    tid = (await client.post("/tickets", json={"text": "help"})).json()["id"]
    ticket = await wait_done(client, tid)
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
    assert claims.count("hello") == 1
    assert claims.count(None) == 9


async def test_healthz(client):
    assert (await client.get("/healthz")).json() == {"status": "ok"}
