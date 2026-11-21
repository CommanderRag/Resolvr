import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://resolvr:resolvr@localhost:5432/resolvr")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Per-client-IP limit on POST /tickets (requests/minute, Redis-backed). 0 disables it.
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "0"))
# Remote Redis (e.g. Upstash over TLS) needs more headroom than localhost.
REDIS_CONNECT_TIMEOUT = float(os.environ.get("REDIS_CONNECT_TIMEOUT", "0.5"))
REDIS_TIMEOUT = float(os.environ.get("REDIS_TIMEOUT", "1.0"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))
# LLM provider limits: cap simultaneous calls, and let the SDK retry 429/5xx with backoff (it honours Retry-After).
MAX_CONCURRENT_LLM = int(os.environ.get("MAX_CONCURRENT_LLM", "4"))
GROQ_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "6"))
# Where the existing Resolvr code and artifacts live (repo-root relative by default).
RESOLVR_SRC = os.environ.get("RESOLVR_SRC", "../src")
DATA_DIR = os.environ.get("DATA_DIR", "../data")
