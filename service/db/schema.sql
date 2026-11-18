-- Resolvr service schema (PostgreSQL 13+, uses built-in gen_random_uuid()).
-- Applied automatically by docker compose on first start of the db volume.

-- Retrieval corpus: the (issue, resolution) pairs from data/grounding_pairs.pkl.
-- `id` is the row's position in that pickle, i.e. its index into the embedding matrix.
CREATE TABLE grounding_pairs (
    id         integer PRIMARY KEY,
    issue      text NOT NULL,
    resolution text NOT NULL
);

CREATE TYPE ticket_status AS ENUM ('received', 'processing', 'done', 'failed');

CREATE TABLE tickets (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    text              text NOT NULL CHECK (length(text) BETWEEN 1 AND 2000),
    text_hash         text NOT NULL CHECK (length(text_hash) = 64),  -- sha256 hex of normalised text: dedupe / cache key (text, not char(n): char(n)=text comparisons can't use the index)
    status            ticket_status NOT NULL DEFAULT 'received',
    intent            text,
    auto_handle       boolean,
    escalation_reason text,
    reply             text,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    processed_at      timestamptz,
    latency_ms        integer,                    -- processing time (classify+retrieve+draft, or cache lookup)
    cache_hit         boolean NOT NULL DEFAULT false,  -- answered from Redis instead of calling the LLM
    -- a finished ticket must carry its result; a failed one must carry an error
    CONSTRAINT done_has_result  CHECK (status <> 'done'   OR (intent IS NOT NULL AND reply IS NOT NULL AND auto_handle IS NOT NULL)),
    CONSTRAINT failed_has_error CHECK (status <> 'failed' OR error IS NOT NULL)
);

-- Which retrieved resolutions grounded a ticket's reply, in rank order.
CREATE TABLE ticket_grounding (
    ticket_id  uuid     NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
    rank       smallint NOT NULL,
    pair_id    integer  NOT NULL REFERENCES grounding_pairs(id),
    similarity real     NOT NULL,
    PRIMARY KEY (ticket_id, rank)
);

-- Work queue: only unfinished tickets are indexed, so it stays tiny as history grows.
CREATE INDEX tickets_queue_idx ON tickets (created_at) WHERE status IN ('received', 'processing');
-- Duplicate / cache lookups by content.
CREATE INDEX tickets_text_hash_idx ON tickets (text_hash);
-- "Recent tickets for intent X" dashboards and analytics.
CREATE INDEX tickets_intent_created_idx ON tickets (intent, created_at DESC) WHERE intent IS NOT NULL;
