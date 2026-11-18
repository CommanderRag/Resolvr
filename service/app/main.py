import hashlib
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import config, db


class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await db.create_pool(config.DATABASE_URL)
    if getattr(app.state, "engine", None) is None:  # tests inject a fake engine instead
        from .engine import Engine

        app.state.engine = Engine(config.RESOLVR_SRC, config.DATA_DIR)
        await db.seed_grounding(app.state.pool, app.state.engine.pairs)
    yield
    await app.state.pool.close()


app = FastAPI(title="Resolvr", lifespan=lifespan)


async def process_ticket(app: FastAPI, ticket_id: uuid.UUID) -> None:
    """Runs after the HTTP response has been sent. In-process for now; the next step
    moves this behind a Redis queue + separate worker so work survives restarts."""
    pool = app.state.pool
    text = await db.claim_ticket(pool, ticket_id)
    if text is None:  # already claimed by someone else
        return
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(app.state.engine.process, text)
        await db.finish_ticket(pool, ticket_id, result, int((time.perf_counter() - started) * 1000))
    except Exception as exc:  # record the failure on the ticket instead of losing it
        await db.fail_ticket(pool, ticket_id, f"{type(exc).__name__}: {exc}"[:500])


@app.post("/tickets", status_code=202)
async def create_ticket(body: TicketIn, request: Request, tasks: BackgroundTasks):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    text_hash = hashlib.sha256(text.lower().encode()).hexdigest()
    row = await db.insert_ticket(request.app.state.pool, text, text_hash)
    tasks.add_task(process_ticket, request.app, row["id"])
    return {"id": str(row["id"]), "status": row["status"], "links": {"self": f"/tickets/{row['id']}"}}


@app.get("/tickets/{ticket_id}")
async def read_ticket(ticket_id: uuid.UUID, request: Request):
    ticket = await db.get_ticket(request.app.state.pool, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return ticket


@app.get("/healthz")
async def healthz(request: Request):
    await request.app.state.pool.fetchval("SELECT 1")
    return {"status": "ok"}
