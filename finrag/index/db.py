"""Postgres + pgvector storage for filings and chunks."""

from __future__ import annotations

from importlib import resources

import psycopg
from pgvector.psycopg import register_vector

from finrag.ingest.chunk import Chunk
from finrag.ingest.edgar import FilingMeta


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int, model: str) -> None:
    sql = resources.files("finrag.index").joinpath("schema.sql").read_text()
    conn.execute(sql.replace("{dim}", str(dim)))
    row = conn.execute("SELECT value FROM index_meta WHERE key = 'embedding'").fetchone()
    want = f"{model}:{dim}"
    if row and row[0] != want:
        raise RuntimeError(
            f"Index was built with {row[0]}, settings say {want}. "
            "Run `finrag db-reset` to rebuild with the new model."
        )
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES ('embedding', %s) ON CONFLICT DO NOTHING",
        (want,),
    )


def reset(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks, filings, index_meta CASCADE")


def upsert_filing(conn: psycopg.Connection, m: FilingMeta) -> None:
    conn.execute(
        """
        INSERT INTO filings (accession, cik, ticker, company, fiscal_year, form,
                             filing_date, report_date, index_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (accession) DO UPDATE SET fiscal_year = EXCLUDED.fiscal_year
        """,
        (m.accession, m.cik, m.ticker, m.company, m.fiscal_year, m.form,
         m.filing_date or None, m.report_date or None, m.index_url),
    )


def upsert_chunks(conn: psycopg.Connection, chunks: list[Chunk]) -> int:
    """Insert or update chunk text. The embedding is cleared only if the content changed."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (id, accession, ticker, company, fiscal_year, document, item,
                                section, page_start, page_end, page_label, chunk_type, content,
                                n_tokens, content_sha, source_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                item = EXCLUDED.item, section = EXCLUDED.section,
                page_start = EXCLUDED.page_start, page_end = EXCLUDED.page_end,
                page_label = EXCLUDED.page_label, chunk_type = EXCLUDED.chunk_type,
                content = EXCLUDED.content, n_tokens = EXCLUDED.n_tokens,
                source_url = EXCLUDED.source_url, content_sha = EXCLUDED.content_sha,
                embedding = CASE WHEN chunks.content_sha = EXCLUDED.content_sha
                                 THEN chunks.embedding END,
                embedding_model = CASE WHEN chunks.content_sha = EXCLUDED.content_sha
                                       THEN chunks.embedding_model END
            -- Leave unchanged rows alone: every rewrite adds an HNSW entry and graph repairs.
            WHERE (chunks.item, chunks.section, chunks.page_start, chunks.page_end,
                   chunks.page_label, chunks.chunk_type, chunks.content_sha, chunks.source_url)
                  IS DISTINCT FROM
                  (EXCLUDED.item, EXCLUDED.section, EXCLUDED.page_start, EXCLUDED.page_end,
                   EXCLUDED.page_label, EXCLUDED.chunk_type, EXCLUDED.content_sha,
                   EXCLUDED.source_url)
            """,
            [
                (c.id, c.accession, c.ticker, c.company, c.fiscal_year, c.document, c.item,
                 c.section, c.page_start, c.page_end, c.page_label, c.chunk_type, c.content,
                 c.n_tokens, c.content_sha, c.source_url)
                for c in chunks
            ],
        )
    return len(chunks)


def rebuild_vector_index(conn: psycopg.Connection) -> None:
    """Build the HNSW graph again over all vectors. A graph grown row by row while vectors
    are written and rewritten finds neighbours worse than one built in one go."""
    conn.execute("REINDEX INDEX chunks_embedding_idx")


def delete_stale_chunks(conn: psycopg.Connection, accession: str, keep_ids: list[str]) -> int:
    """Remove chunks of a filing that a re-parse no longer produces."""
    cur = conn.execute(
        "DELETE FROM chunks WHERE accession = %s AND NOT (id = ANY(%s))", (accession, keep_ids)
    )
    return cur.rowcount
