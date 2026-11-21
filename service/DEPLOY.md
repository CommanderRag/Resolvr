# Deploying Resolvr (no credit card)

API on a Hugging Face Docker Space, Postgres on Neon, Redis on Upstash.
Free-tier limits change, so check each provider's current terms.

## 1. Postgres (Neon)

1. Create a project and copy the connection string. It ends in `?sslmode=require`.
2. Apply the schema once (the app only applies small upgrades at startup, not the full schema):

   ```bash
   psql "<neon-connection-string>" -f service/db/schema.sql
   ```

## 2. Redis (Upstash)

Create a database and copy the TLS URL: `rediss://default:<password>@<host>:6379`.

## 3. Space

1. huggingface.co → New Space → SDK **Docker** → blank template.
2. In the Space's *Settings → Variables and secrets*, add:

   | Name | Type | Value |
   |---|---|---|
   | `GROQ_API_KEY` | secret | your key |
   | `DATABASE_URL` | secret | Neon string |
   | `REDIS_URL` | secret | Upstash `rediss://` URL |
   | `RATE_LIMIT_PER_MIN` | variable | `30` |
   | `REDIS_CONNECT_TIMEOUT` | variable | `3` |
   | `REDIS_TIMEOUT` | variable | `3` |
   | `HF_HOME` | variable | `/tmp/hf` |

   The default 0.5 s Redis connect timeout is too tight for a remote TLS Redis and would make
   every request a cache miss. `HF_HOME` avoids permission errors if the container is not root.

3. Push the code. The Space needs a README with front matter, so use a throwaway branch and
   leave your GitHub README untouched:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/resolvr
   git checkout -b hf
   printf -- '---\ntitle: Resolvr\nsdk: docker\napp_port: 8000\n---\n' | cat - README.md > R.md && mv R.md README.md
   git commit -am "Space front matter"
   git push space hf:main --force
   git checkout main
   ```

   (Use a Hugging Face access token with write access as the password.)

4. The first build is slow (torch + model download). Watch the Space's *Logs* tab.

## 4. Check it

```bash
URL=https://<user>-resolvr.hf.space
curl $URL/healthz                       # {"status":"ok","redis":"up"}
curl -X POST $URL/tickets -H 'content-type: application/json' -d '{"text":"where is my order?"}'
curl $URL/tickets/<id>
curl $URL/stats                         # cache hit rate after repeating a message
```

## Known caveats

- Free Spaces sleep when idle, so the first request after a pause is slow.
- Rate limiting identifies callers by the last `X-Forwarded-For` entry, which is correct behind a
  single trusted proxy. Check the Space logs for the IPs it sees.
- Run the load test against your local compose stack, not the public URL: the public one has
  rate limiting and remote Redis/Postgres, so the numbers aren't comparable.
- Unfinished tickets are lost if the container restarts, because processing is still in-process.
