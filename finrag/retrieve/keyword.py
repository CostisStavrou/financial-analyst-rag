"""Keyword retrieval: BM25 in plain SQL over the keyword_docs / keyword_postings views.

Plain SQL needs no extension beyond pgvector, so it runs the same on local Postgres, Cloud SQL
and AlloyDB. Lucene's BM25, with its defaults:

    score = sum over the question's terms t of
            qtf * idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

- Terms: Postgres's english text search (stemmed, stop words dropped), for the question and
  the chunks alike. qtf: how often the term is in the question.
- tf: the term's positions in the chunk. dl: all positions in the chunk.
- N, df and avgdl cover the whole corpus, also when filtered, so a filtered ranking is the
  unfiltered one restricted to the filter. Lucene's idf stays positive for terms in more than
  half the chunks ("item", "statement"); the classic form would turn negative there.
- A chunk matches if it has any of the question's terms. plainto_tsquery and
  websearch_to_tsquery want every word and match no chunk for a typical question.
- Ties: scores are summed as exact decimals and rounded to 9 places, then ordered by md5(id),
  which favours no company or year (ids start with ticker and fiscal year).
"""

from __future__ import annotations

import psycopg

from finrag.retrieve.vector import COLUMNS, Hit

K1, B = 1.2, 0.75

# views not made yet (database from before them) or not filled yet
MISSING = (psycopg.errors.UndefinedTable, psycopg.errors.ObjectNotInPrerequisiteState)
NOT_BUILT = "The keyword index is missing or empty. Run `finrag index`."


def _query(question: str, k: int, ticker: str | None, fiscal_year: int | None,
           item: str | None) -> tuple[str, dict]:
    where = []
    if ticker:
        where.append("d.ticker = %(ticker)s")
    if fiscal_year:
        where.append("d.fiscal_year = %(fy)s")
    if item:
        where.append("d.item = %(item)s")
    sql = f"""
        WITH q AS MATERIALIZED (  -- the question's terms, with their count in it
            SELECT lexeme, cardinality(positions) AS qtf
            FROM unnest(to_tsvector('english', %(question)s))
        ),
        corpus AS MATERIALIZED (SELECT count(*) AS n, avg(dl) AS avgdl FROM keyword_docs),
        terms AS MATERIALIZED (  -- document frequency over the whole corpus
            SELECT q.lexeme, q.qtf, count(*) AS df
            FROM q JOIN keyword_postings p USING (lexeme)
            GROUP BY q.lexeme, q.qtf
        ),
        scored AS (
            SELECT p.id, round(sum(
                       t.qtf * ln(1 + (c.n - t.df + 0.5) / (t.df + 0.5))
                       * p.tf * ({K1} + 1) / (p.tf + {K1} * (1 - {B} + {B} * d.dl / c.avgdl))
                   ), 9) AS score
            FROM terms t
            JOIN keyword_postings p USING (lexeme)
            JOIN keyword_docs d USING (id)
            CROSS JOIN corpus c
            {"WHERE " + " AND ".join(where) if where else ""}
            GROUP BY p.id
            ORDER BY score DESC, md5(p.id)
            LIMIT %(k)s
        )
        SELECT {COLUMNS}, s.score::float8
        FROM scored s JOIN chunks USING (id)
        ORDER BY s.score DESC, md5(s.id)
    """
    params = {"question": question, "ticker": ticker, "fy": fiscal_year, "item": item, "k": k}
    return sql, params


def search(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
) -> list[Hit]:
    """Top k chunks by BM25 score. No hits if the question has no searchable term."""
    sql, params = _query(question, k, ticker, fiscal_year, item)
    try:
        with conn.transaction(force_rollback=True):
            # plan each search with its own filter values, as vector search does
            conn.execute("SET LOCAL plan_cache_mode = force_custom_plan")
            rows = conn.execute(sql, params).fetchall()
    except MISSING as e:
        raise RuntimeError(NOT_BUILT) from e
    return [Hit(*r[:-1], score=float(r[-1])) for r in rows]


def corpus_stats(conn: psycopg.Connection) -> dict:
    """N, avgdl and number of distinct terms in the keyword index."""
    n, avgdl = conn.execute("SELECT count(*), avg(dl) FROM keyword_docs").fetchone()
    terms = conn.execute("SELECT count(DISTINCT lexeme) FROM keyword_postings").fetchone()[0]
    return {"chunks": n, "avgdl": round(float(avgdl), 2), "terms": terms}
