"""Redis cache of finished results, keyed by the hash of the normalised message.

Redis is an optimisation, never a dependency: every operation swallows connection
errors and behaves as a miss, so the API keeps working (just slower) if Redis is down.
"""
import json
import logging
import uuid
from typing import Optional, Tuple

import redis.asyncio as redis

from .engine import Result

log = logging.getLogger("resolvr.cache")

RESULT_KEY = "resolvr:v1:result:{}"
LOCK_KEY = "resolvr:v1:lock:{}"

# Delete the lock only if we still own it (it may have expired and been re-acquired).
_RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

_ERRORS = (redis.RedisError, OSError)


class Cache:
    def __init__(self, url: str, ttl: int = 3600, lock_ttl: int = 60):
        self._r = redis.from_url(url, socket_connect_timeout=0.5, socket_timeout=1.0)
        self._ttl = ttl
        self._lock_ttl = lock_ttl

    async def get(self, text_hash: str) -> Optional[Result]:
        try:
            raw = await self._r.get(RESULT_KEY.format(text_hash))
        except _ERRORS as exc:
            log.warning("cache get failed, treating as miss: %s", exc)
            return None
        if raw is None:
            return None
        d = json.loads(raw)
        return Result(d["intent"], d["auto_handle"], d["reason"], d["reply"],
                      [(int(p), float(s)) for p, s in d["grounding"]])

    async def set(self, text_hash: str, result: Result) -> None:
        payload = json.dumps({
            "intent": result.intent, "auto_handle": result.auto_handle, "reason": result.reason,
            "reply": result.reply, "grounding": [list(g) for g in result.grounding],
        })
        try:
            await self._r.set(RESULT_KEY.format(text_hash), payload, ex=self._ttl)
        except _ERRORS as exc:
            log.warning("cache set failed: %s", exc)

    async def acquire_lock(self, text_hash: str) -> Tuple[bool, str]:
        """Single-flight lock so identical concurrent messages cost ONE LLM call.
        Returns (acquired, token). If Redis is down we report acquired=True so work is not blocked."""
        token = uuid.uuid4().hex
        try:
            ok = await self._r.set(LOCK_KEY.format(text_hash), token, nx=True, ex=self._lock_ttl)
            return bool(ok), token
        except _ERRORS as exc:
            log.warning("cache lock failed, proceeding without it: %s", exc)
            return True, token

    async def release_lock(self, text_hash: str, token: str) -> None:
        try:
            await self._r.eval(_RELEASE_LOCK, 1, LOCK_KEY.format(text_hash), token)
        except _ERRORS as exc:
            log.warning("cache unlock failed: %s", exc)

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except _ERRORS:
            return False

    async def clear(self) -> None:
        """Flush the selected Redis database. Tests only."""
        try:
            await self._r.flushdb()
        except _ERRORS:
            pass

    async def close(self) -> None:
        await self._r.aclose()
