"""All SQL lives here. Plain asyncpg + hand-written SQL on purpose."""
import uuid
from typing import Optional

import asyncpg


async def create_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=2, max_size=10)


async def seed_grounding(pool: asyncpg.Pool, pairs: list) -> int:
    """Upsert the retrieval corpus. Runs at every startup, so it also repairs a
    missing or partially-filled table (idempotent, no truncation)."""
    async with pool.acquire() as conn:
        await conn.executemany(
            """INSERT INTO grounding_pairs (id, issue, resolution) VALUES ($1, $2, $3)
               ON CONFLICT (id) DO UPDATE SET issue = EXCLUDED.issue, resolution = EXCLUDED.resolution""",
            [(i, p["issue"], p["resolution"]) for i, p in enumerate(pairs)],
        )
    return len(pairs)


async def insert_ticket(pool: asyncpg.Pool, text: str, text_hash: str) -> asyncpg.Record:
    return await pool.fetchrow(
        "INSERT INTO tickets (text, text_hash) VALUES ($1, $2) RETURNING id, status, created_at",
        text,
        text_hash,
    )


async def claim_ticket(pool: asyncpg.Pool, ticket_id: uuid.UUID) -> Optional[str]:
    """Atomically move received -> processing. Returns the ticket text, or None if
    someone else already claimed it. The same statement works for a multi-worker queue."""
    return await pool.fetchval(
        "UPDATE tickets SET status = 'processing' WHERE id = $1 AND status = 'received' RETURNING text",
        ticket_id,
    )


async def finish_ticket(pool: asyncpg.Pool, ticket_id: uuid.UUID, result, latency_ms: int) -> None:
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """UPDATE tickets
                      SET status = 'done', intent = $2, auto_handle = $3, escalation_reason = $4,
                          reply = $5, processed_at = now(), latency_ms = $6
                    WHERE id = $1""",
                ticket_id, result.intent, result.auto_handle, result.reason, result.reply, latency_ms,
            )
            await conn.executemany(
                "INSERT INTO ticket_grounding (ticket_id, rank, pair_id, similarity) VALUES ($1, $2, $3, $4)",
                [(ticket_id, rank, pair_id, sim) for rank, (pair_id, sim) in enumerate(result.grounding, start=1)],
            )


async def fail_ticket(pool: asyncpg.Pool, ticket_id: uuid.UUID, error: str) -> None:
    await pool.execute(
        "UPDATE tickets SET status = 'failed', error = $2, processed_at = now() WHERE id = $1",
        ticket_id,
        error,
    )


async def get_ticket(pool: asyncpg.Pool, ticket_id: uuid.UUID) -> Optional[dict]:
    row = await pool.fetchrow(
        """SELECT id, text, status, intent, auto_handle, escalation_reason, reply, error,
                  created_at, processed_at, latency_ms
             FROM tickets WHERE id = $1""",
        ticket_id,
    )
    if row is None:
        return None
    grounding = await pool.fetch(
        """SELECT g.rank, p.issue, p.resolution, g.similarity
             FROM ticket_grounding g JOIN grounding_pairs p ON p.id = g.pair_id
            WHERE g.ticket_id = $1 ORDER BY g.rank""",
        ticket_id,
    )
    ticket = dict(row)
    ticket["grounding"] = [dict(g) for g in grounding]
    return ticket
