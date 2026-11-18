# Resolvr service

HTTP API around the Resolvr pipeline: submit a support message, get back its intent,
a grounded reply draft and an escalation decision.

```
POST /tickets        -> 202 {id, status:"received"}   (processing happens after the response)
GET  /tickets/{id}   -> status received|processing|done|failed, result, grounding
GET  /stats          -> counts, cache hit rate, avg processing ms on hit vs miss
GET  /healthz        -> db + redis status
```

## Run

```bash
cp .env.example .env          # add GROQ_API_KEY
docker compose up --build     # db + api on :8000

curl -s -X POST localhost:8000/tickets -H 'content-type: application/json' \
     -d '{"text": "my order has not arrived and it was due Monday"}'
curl -s localhost:8000/tickets/<id>
```

## Test

```bash
docker compose up -d db redis
cd service && pip install -r requirements-dev.txt
TEST_DATABASE_URL=postgresql://resolvr:resolvr@localhost:5432/resolvr_test pytest -q   # DB must end in _test; created automatically
```

## Design notes

- **202 + polling** instead of holding the request open for LLM calls (seconds).
- **Atomic claim** (`UPDATE ... WHERE status='received' RETURNING`) so a second worker can never
  double-process a ticket. The same statement is what a Redis-fed worker pool will use.
- **Blocking work off the event loop**: the pipeline runs in a thread pool.
- **Partial index** on unfinished tickets keeps the queue lookup tiny as history grows.
- **Failures are data**: an engine error marks the ticket `failed` with the error text.
- **Cache-aside on Redis**, keyed by sha256 of the lower-cased, trimmed message (1 h TTL). A repeat
  message skips the LLM entirely; each ticket records `cache_hit` so `/stats` can show the hit rate and
  the latency difference.
- **Single-flight**: identical concurrent messages take a Redis lock (`SET NX EX`, released with a
  compare-and-delete Lua script); one request calls the LLM, the others wait for its result.
- **Redis is an optimisation, not a dependency**: if it is down every cache call behaves as a miss and
  the API keeps working, just slower. Failures are never cached.
- **Provider rate limits**: a burst of never-seen messages used to fail with HTTP 429 (4 of 15 in a load
  test). Engine calls are now capped by a semaphore (`MAX_CONCURRENT_LLM`, default 4) so bursts queue,
  and the Groq client retries 429/5xx with exponential backoff (`GROQ_MAX_RETRIES`, default 6).
