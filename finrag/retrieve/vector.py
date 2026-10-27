"""Dense retrieval: cosine similarity over pgvector, with optional metadata filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg


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


def search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    model: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
) -> list[Hit]:
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
    with conn.transaction():
        # With filters, let the HNSW scan keep going until k rows match (pgvector >= 0.8).
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        rows = conn.execute(sql, params).fetchall()
    hits = [Hit(*r[:-1], score=float(r[-1])) for r in rows]
    return sorted(hits, key=lambda h: h.score, reverse=True)  # relaxed_order may shuffle
