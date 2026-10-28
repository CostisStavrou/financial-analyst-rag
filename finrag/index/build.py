"""Load parsed chunks into Postgres and embed the ones that have no current vector."""

from __future__ import annotations

import logging
import time

from finrag.config import Settings
from finrag.index import db
from finrag.index.embed import make_embedder
from finrag.ingest.build import load_chunks
from finrag.ingest.edgar import iter_downloaded

log = logging.getLogger(__name__)


def build_index(
    settings: Settings, tickers: set[str] | None = None, limit: int | None = None,
    batch: int = 200, first: list[tuple[str, int]] | None = None, reindex: bool = False,
) -> dict:
    """`first`: (ticker, fiscal year) filings to embed before the rest, in that order.
    The keyword index is rebuilt every run. The HNSW index is rebuilt after new vectors are
    written, or always with `reindex`."""
    conn = db.connect(settings.database_url)
    db.init_schema(conn, settings.embed_dim, settings.embed_model)

    metas = [m for _, m in iter_downloaded(settings.raw_dir) if not tickers or m.ticker in tickers]
    chunks = load_chunks(settings.parsed_dir, tickers)
    if not chunks:
        raise RuntimeError("No parsed chunks found. Run `finrag parse` first.")
    for m in metas:
        db.upsert_filing(conn, m)
    db.upsert_chunks(conn, chunks)
    stale = 0
    by_acc: dict[str, list[str]] = {}
    for c in chunks:
        by_acc.setdefault(c.accession, []).append(c.id)
    for acc, ids in by_acc.items():
        stale += db.delete_stale_chunks(conn, acc, ids)
    # before embedding, which can stop on a quota error; the keyword index needs no vectors
    t0 = time.perf_counter()
    db.refresh_keyword_index(conn)
    keyword_refresh = time.perf_counter() - t0
    log.info("keyword index rebuilt in %.1fs", keyword_refresh)

    todo_ids = {
        r[0]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s",
            (settings.embed_model,),
        )
    }
    todo = [c for c in chunks if c.id in todo_ids]
    if first:
        rank = {f: n for n, f in enumerate(first)}
        todo.sort(key=lambda c: rank.get((c.ticker, c.fiscal_year), len(rank)))
    if limit is not None:
        todo = todo[:limit]
    log.info("%d chunks loaded, %d need embeddings (%s)", len(chunks), len(todo),
             settings.embed_model)

    embedder = make_embedder(settings) if todo else None
    t0 = time.perf_counter()
    done = 0
    for i in range(0, len(todo), batch):
        part = todo[i : i + batch]
        vecs = embedder.embed_documents([c.embed_text() for c in part])
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding = %s, embedding_model = %s WHERE id = %s",
                [(v, settings.embed_model, c.id) for v, c in zip(vecs, part)],
            )
        done += len(part)
        impl = embedder.impl
        log.info("embedded %d/%d (requests %d, rate-limited %d, %.0fs)", done, len(todo),
                 impl.requests, getattr(impl, "rate_limited", 0), time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    rebuild = 0.0
    if done or reindex:
        t0 = time.perf_counter()
        db.rebuild_vector_index(conn)
        rebuild = time.perf_counter() - t0
        log.info("HNSW index rebuilt in %.1fs", rebuild)

    total, embedded = conn.execute(
        "SELECT count(*), count(embedding) FROM chunks"
    ).fetchone()
    return {
        "chunks_in_files": len(chunks),
        "stale_removed": stale,
        "embedded_now": done,
        "cache_hits": embedder.cache_hits if embedder else 0,
        "api_requests": embedder.impl.requests if embedder else 0,
        "rate_limited": getattr(embedder.impl, "rate_limited", 0) if embedder else 0,
        "est_tokens_sent": embedder.impl.tokens_sent if embedder else 0,
        "embed_seconds": round(elapsed, 1),
        "index_rebuild_seconds": round(rebuild, 1),
        "keyword_refresh_seconds": round(keyword_refresh, 1),
        "chunks_in_db": total,
        "chunks_with_embedding": embedded,
    }
