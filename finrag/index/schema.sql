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
    embedding       vector({dim})
);

CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (ticker, fiscal_year);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Content-only full-text column from the first schema, replaced by keyword_docs below.
-- Dropping a column doesn't rewrite the table, so the HNSW graph stays as it is.
ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;

-- Keyword (BM25) index, filled by `finrag index`. The text is exactly what the embedding
-- model sees (Chunk.embed_text: a header with company, filing, section and page, then the
-- content), so both retrievers search the same text. Materialised views rather than chunks
-- columns: changing them never rewrites chunks or rebuilds its HNSW index.
CREATE MATERIALIZED VIEW IF NOT EXISTS keyword_docs AS
SELECT c.id, c.ticker, c.fiscal_year, c.item, md5(t.text) AS text_md5, v.tsv,
       coalesce((SELECT sum(cardinality(u.positions)) FROM unnest(v.tsv) u), 0)::int AS dl
FROM chunks c,
     LATERAL (SELECT c.company || ' (' || c.ticker || ') '
                     || CASE WHEN c.document = '10-K' THEN '10-K' ELSE '10-K ' || c.document END
                     || ', fiscal year ' || c.fiscal_year || '. ' || c.section
                     || '. Page ' || c.page_start || E'.\n\n' || c.content AS text) t,
     LATERAL (SELECT to_tsvector('english', t.text) AS tsv) v
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS keyword_docs_id_idx ON keyword_docs (id);
CREATE INDEX IF NOT EXISTS keyword_docs_filter_idx ON keyword_docs (ticker, fiscal_year);

-- Inverted index: one row per (term, chunk), tf = the term's positions in the chunk.
CREATE MATERIALIZED VIEW IF NOT EXISTS keyword_postings AS
SELECT u.lexeme, d.id, cardinality(u.positions) AS tf
FROM keyword_docs d, unnest(d.tsv) u
WITH NO DATA;

CREATE INDEX IF NOT EXISTS keyword_postings_lexeme_idx ON keyword_postings (lexeme);
