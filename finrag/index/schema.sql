-- Postgres + pgvector schema. {dim} is filled in from EMBED_DIM.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS index_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS filings (
    accession    text PRIMARY KEY,
    cik          integer NOT NULL,
    ticker       text NOT NULL,
    company      text NOT NULL,
    fiscal_year  integer NOT NULL,
    form         text NOT NULL,
    filing_date  date,
    report_date  date,
    index_url    text
);

CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    accession       text NOT NULL REFERENCES filings(accession) ON DELETE CASCADE,
    ticker          text NOT NULL,
    company         text NOT NULL,
    fiscal_year     integer NOT NULL,
    document        text NOT NULL,
    item            text NOT NULL,
    section         text NOT NULL,
    page_start      integer NOT NULL,
    page_end        integer NOT NULL,
    page_label      text,
    chunk_type      text NOT NULL,
    content         text NOT NULL,
    n_tokens        integer NOT NULL,
    content_sha     text NOT NULL,
    source_url      text NOT NULL,
    embedding_model text,
    embedding       vector({dim}),
    -- full-text search, for hybrid retrieval (days 5-8)
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (ticker, fiscal_year);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
