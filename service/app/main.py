import asyncio
import hashlib
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import config, db
from .cache import Cache
from .resolver import resolve


class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await db.create_pool(config.DATABASE_URL)
    await db.apply_upgrades(app.state.pool)
    if getattr(app.state, "cache", None) is None:  # tests inject their own
        app.state.cache = Cache(config.REDIS_URL, config.CACHE_TTL_SECONDS,
                                     connect_timeout=config.REDIS_CONNECT_TIMEOUT, timeout=config.REDIS_TIMEOUT)
    if getattr(app.state, "llm_slots", None) is None:
        app.state.llm_slots = asyncio.Semaphore(config.MAX_CONCURRENT_LLM)
    if getattr(app.state, "engine", None) is None:
        from .engine import Engine

        app.state.engine = Engine(config.RESOLVR_SRC, config.DATA_DIR, config.GROQ_MAX_RETRIES)
        await db.seed_grounding(app.state.pool, app.state.engine.pairs)
    yield
    await app.state.cache.close()
    await app.state.pool.close()


app = FastAPI(title="Resolvr", lifespan=lifespan)


async def process_ticket(app: FastAPI, ticket_id: uuid.UUID) -> None:
    """Runs after the HTTP response has been sent. In-process for now; the next step
    moves this behind a queue + separate worker so work survives restarts."""
    pool = app.state.pool
    claimed = await db.claim_ticket(pool, ticket_id)
    if claimed is None:  # already claimed by someone else
        return
    started = time.perf_counter()
    try:
        result, cache_hit = await resolve(
            app.state.engine, app.state.cache, claimed["text"], claimed["text_hash"],
            slots=getattr(app.state, "llm_slots", None),
        )
        await db.finish_ticket(pool, ticket_id, result, int((time.perf_counter() - started) * 1000), cache_hit)
    except Exception as exc:  # record the failure on the ticket instead of losing it
        await db.fail_ticket(pool, ticket_id, f"{type(exc).__name__}: {exc}"[:500])


def client_id(request: Request) -> str:
    """Identify the caller for rate limiting. Behind a proxy the *last* X-Forwarded-For entry is
    the one the proxy itself appended (earlier entries are client-supplied and spoofable)."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.post("/tickets", status_code=202)
async def create_ticket(body: TicketIn, request: Request, tasks: BackgroundTasks):
    if config.RATE_LIMIT_PER_MIN > 0 and not await request.app.state.cache.allow(
            client_id(request), config.RATE_LIMIT_PER_MIN):
        raise HTTPException(status_code=429, detail="rate limit exceeded",
                            headers={"Retry-After": "60"})
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    text_hash = hashlib.sha256(text.lower().encode()).hexdigest()  # same message, any casing -> same key
    row = await db.insert_ticket(request.app.state.pool, text, text_hash)
    tasks.add_task(process_ticket, request.app, row["id"])
    return {"id": str(row["id"]), "status": row["status"], "links": {"self": f"/tickets/{row['id']}"}}


@app.get("/tickets/{ticket_id}")
async def read_ticket(ticket_id: uuid.UUID, request: Request):
    ticket = await db.get_ticket(request.app.state.pool, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return ticket


@app.get("/stats")
async def stats(request: Request):
    """Ticket counts and cache effectiveness (avg processing time on hits vs misses)."""
    return await db.get_stats(request.app.state.pool)


@app.get("/healthz")
async def healthz(request: Request):
    await request.app.state.pool.fetchval("SELECT 1")
    return {"status": "ok", "redis": "up" if await request.app.state.cache.ping() else "down"}
