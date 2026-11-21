![CI](https://github.com/CommanderRag/Resolvr/actions/workflows/ci.yml/badge.svg)
# Resolvr

An AI customer-support service. Send it a customer message; it classifies the intent, retrieves how
the brand actually resolved similar issues, drafts a grounded reply, and decides whether a human
should take over.

Built on the AmazonHelp slice of the [Twitter Customer Support dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
**FastAPI · PostgreSQL · Redis · Docker Compose · PyTest**

## How a request flows

```
client ── POST /tickets ──▶ FastAPI ── INSERT ──▶ Postgres (ticket: received)
   ◀──────── 202 {id} ──────┘
                            └─ background task
                                 │ claim: UPDATE ... WHERE status='received' RETURNING   (atomic)
                                 ▼
                          Redis cache hit? ── yes ──▶ finish ticket (cache_hit = true)
                                 │ no
                          single-flight lock (SET NX EX): identical concurrent messages wait
                                 ▼
                   classify ─▶ retrieve top-3 resolutions (MiniLM) ─▶ draft reply
                   (LLM calls capped + retried with backoff)
                                 ▼
                   cache result · finish ticket + grounding rows in one transaction
client ── GET /tickets/{id} ──▶ status, intent, reply, escalation decision, grounding
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /tickets` `{"text": "..."}` | Accepts a message, returns `202 {id}` immediately |
| `GET /tickets/{id}` | `received / processing / done / failed`, result, retrieved grounding, `cache_hit` |
| `GET /stats` | Counts, cache hit rate, average processing ms on hit vs miss |
| `GET /healthz` | Database and Redis status |

## Performance

Measured on a single laptop with the load generator, API (one uvicorn worker), Postgres and Redis all
on the same machine, so treat these as relative numbers, not capacity planning. Reproduce with
`python3 service/bench_api.py`.

| Scenario | Result |
|---|---|
| 1,000 requests, 50 concurrent, repeat message | 1000/1000 accepted · p50 58 ms · p95 181 ms · p99 284 ms · 675 req/s |
| 3,000 requests, 100 concurrent, repeat message | 0 errors · p50 120 ms · p95 433 ms · 659 req/s accepted · 630 tickets/s fully processed |
| Processing time, cache hit vs miss | ~1-2 ms vs ~2.2 s (a miss makes two LLM calls) |
| Burst of 15 never-seen messages | **before:** 4 failed (provider rate limit, HTTP 429) · **after** concurrency cap + retries: 0 failed, at the cost of a longer drain (12 s → 26 s) because requests now queue and back off instead of failing |

## Design decisions

- **202 + polling.** LLM calls take seconds; holding the request open would cap concurrency and tie up connections.
- **Atomic claim.** `UPDATE ... WHERE status='received' RETURNING` guarantees a ticket is processed once, even with several workers racing. A test fires 10 concurrent claims and asserts exactly one wins.
- **Cache-aside with single-flight.** Key = SHA-256 of the lower-cased, trimmed message. Identical concurrent messages take a Redis lock (`SET NX EX`, released by a Lua compare-and-delete script); one request calls the LLM and the rest wait for its result. If the lock holder fails, a waiter takes over.
- **Redis is an optimisation, not a dependency.** Any Redis error behaves as a cache miss, so an outage makes the API slower, not broken. Failures are never cached.
- **Provider protection.** An `asyncio.Semaphore` caps simultaneous LLM calls (`MAX_CONCURRENT_LLM`), and the client retries 429/5xx with exponential backoff (`GROQ_MAX_RETRIES`).
- **Blocking work off the event loop.** The pipeline runs in a thread pool.
- **Schema.** Partial index over unfinished tickets (the work queue stays small as history grows), CHECK constraints (a `done` ticket must carry a result, a `failed` one an error), and an FK-linked `ticket_grounding` table recording which past resolutions produced each reply.
- **Tests run against real Postgres and Redis**, not mocks (18 tests, including concurrency, failure and Redis-down cases). They refuse to run on any database not named `*_test`.

## Run it

```bash
cp .env.example .env              # add GROQ_API_KEY (free tier: console.groq.com)
docker compose up --build         # Postgres + Redis + API on :8000

curl -s -X POST localhost:8000/tickets -H 'content-type: application/json' \
     -d '{"text": "my order has not arrived and it was due Monday"}'
curl -s localhost:8000/tickets/<id>
curl -s localhost:8000/stats
```

Tests:

```bash
docker compose up -d db redis
cd service && pip install -r requirements-dev.txt
TEST_DATABASE_URL=postgresql://resolvr:resolvr@localhost:5432/resolvr_test pytest -q
```

## Pipeline quality

Evaluated offline on 188 hand-labelled messages (`python src/evaluate.py`); full write-up in [report.md](report.md).

| System | Intent accuracy |
|---|---|
| Always predict the most common intent | 27.1% |
| TF-IDF + logistic regression | ~34% |
| Prompted LLM (this pipeline) | **51.6%** (95% CI 44-59%) |

Escalation decisions are 84.6% accurate against 76.1% for "always escalate" (96.5% recall, 85.2% precision). On 30 hand-scored replies the LLM-as-judge agreed poorly with human scoring (r ≈ 0 on grounding and correctness), so its reply-quality scores are reported but not relied on.

## Known limitations

- Background processing is in-process: tickets in flight when the API restarts are not recovered. Next step: a durable queue and separate workers (the atomic claim already supports multiple workers).
- Escalation is an intent-level policy, not a per-message judgement.
- Drafted replies sometimes contain template placeholders or copied links from the retrieved examples; a post-check is planned.
- The evaluation set is small (188), so accuracy has a wide confidence interval.
