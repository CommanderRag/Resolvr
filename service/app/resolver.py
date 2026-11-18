"""Cache-aside with single-flight around the slow, expensive engine call.

- hit  -> return the cached result immediately (no LLM call)
- miss -> one request takes a Redis lock and computes; identical concurrent requests wait
          for that result instead of each paying for their own LLM call
- if the lock holder fails, a waiter takes over; if Redis is down, everyone just computes
"""
import asyncio
import time

from starlette.concurrency import run_in_threadpool

POLL_INTERVAL = 0.2   # seconds between checks while another request is computing
MAX_WAIT = 20.0       # then give up waiting and compute ourselves


async def _compute(engine, cache, text, text_hash):
    result = await run_in_threadpool(engine.process, text)  # blocking work stays off the event loop
    await cache.set(text_hash, result)
    return result


async def resolve(engine, cache, text: str, text_hash: str):
    """Return (result, cache_hit)."""
    deadline = time.monotonic() + MAX_WAIT
    while True:
        hit = await cache.get(text_hash)
        if hit is not None:
            return hit, True

        acquired, token = await cache.acquire_lock(text_hash)
        if acquired:
            try:
                hit = await cache.get(text_hash)  # it may have landed while we took the lock
                if hit is not None:
                    return hit, True
                return await _compute(engine, cache, text, text_hash), False
            finally:
                await cache.release_lock(text_hash, token)

        if time.monotonic() >= deadline:
            return await _compute(engine, cache, text, text_hash), False
        await asyncio.sleep(POLL_INTERVAL)
