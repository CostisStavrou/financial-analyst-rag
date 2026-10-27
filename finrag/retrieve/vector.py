"""Dense retrieval: cosine similarity over pgvector, with optional metadata filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

EF_SEARCH = 100  # HNSW candidate list size per query


@dataclass
class Hit:
    id: str
    ticker: str
    company: str
    fiscal_year: int
    document: str
    section: str
    page_start: int
    page_end: int
    page_label: str | None
    chunk_type: str
    content: str
    source_url: str
    score: float

    def citation(self) -> str:
        pages = f"p. {self.page_start}" + (
            f"-{self.page_end}" if self.page_end != self.page_start else ""
        )
        if self.page_label:
            pages += f" (printed page {self.page_label})"
        doc = "10-K" if self.document == "10-K" else f"10-K {self.document}"
        return f"{self.company} {doc} FY{self.fiscal_year}, {self.section}, {pages}"


def _query(query_vec: np.ndarray, model: str, k: int, ticker: str | None,
           fiscal_year: int | None, item: str | None) -> tuple[str, dict]:
    where = ["embedding IS NOT NULL", "embedding_model = %(model)s"]
    if ticker:
        where.append("ticker = %(ticker)s")
    if fiscal_year:
        where.append("fiscal_year = %(fy)s")
    if item:
        where.append("item = %(item)s")
    sql = f"""
        SELECT id, ticker, company, fiscal_year, document, section, page_start, page_end,
               page_label, chunk_type, content, source_url,
               1 - (embedding <=> %(q)s) AS score
        FROM chunks
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """
    params = {"q": query_vec, "model": model, "ticker": ticker, "fy": fiscal_year,
              "item": item, "k": k}
    return sql, params


def _set_mode(conn: psycopg.Connection, exact: bool) -> None:
    """Search settings for one block (rolled back afterwards, so they can't leak)."""
    # Plan each search with its own values, as EXPLAIN in uses_hnsw() does: after five runs
    # psycopg prepares the statement, and a cached generic plan could differ.
    conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
    if exact:
        # No index scans: Postgres reads every matching row and sorts by distance.
        conn.execute("SET LOCAL enable_indexscan = off")
    else:
        # With filters, let the HNSW scan keep going until k rows match (pgvector >= 0.8).
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")


def search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    model: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    exact: bool = False,
) -> list[Hit]:
    """Top k chunks by cosine similarity. `exact` skips the HNSW index and compares the query
    with every matching row, to measure what the approximate index loses."""
    sql, params = _query(query_vec, model, k, ticker, fiscal_year, item)
    # Rolled back, not released: inside a caller's transaction SET LOCAL would otherwise last
    # until the caller commits.
    with conn.transaction(force_rollback=True):
        _set_mode(conn, exact)
        rows = conn.execute(sql, params).fetchall()
    hits = [Hit(*r[:-1], score=float(r[-1])) for r in rows]
    return sorted(hits, key=lambda h: h.score, reverse=True)  # relaxed_order may shuffle


def uses_hnsw(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    model: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
) -> bool:
    """Whether Postgres answers this (non-exact) search from the HNSW index. With a selective
    filter the planner may read just the filter's rows and sort them, which is exact."""
    sql, params = _query(query_vec, model, k, ticker, fiscal_year, item)
    with conn.transaction(force_rollback=True):
        _set_mode(conn, exact=False)
        plan = conn.execute(f"EXPLAIN {sql}", params).fetchall()
    return any("chunks_embedding_idx" in row[0] for row in plan)
