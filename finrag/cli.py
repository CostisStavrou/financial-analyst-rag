"""finrag command line: download -> parse -> index -> ask."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from finrag.config import get_settings, load_companies


def _tickers(args) -> set[str] | None:
    return {t.upper() for t in args.tickers} if getattr(args, "tickers", None) else None


def cmd_download(args) -> None:
    from finrag.ingest.edgar import SecClient, download_company

    s = get_settings()
    client = SecClient(s.sec_user_agent)
    wanted = _tickers(args)
    n = 0
    for company in load_companies():
        if wanted and company.ticker not in wanted:
            continue
        n += len(download_company(client, company, s.raw_dir, dry_run=args.dry_run))
    print(f"{n} filings {'found' if args.dry_run else 'ready'} in {s.raw_dir} "
          f"({client.requests_made} SEC requests)")


def cmd_parse(args) -> None:
    from finrag.ingest.build import build_chunks

    s = get_settings()
    rows = build_chunks(s.raw_dir, s.parsed_dir, _tickers(args))
    print(f"{'filing':<14}{'pages':>7}{'chunks':>8}{'tables':>8}{'~tokens':>10}  sections found")
    for r in rows:
        pages = sum(r["pages"].values())
        secs = len(r["sections"].get("10-K", []))
        extra = " +EX-13" if len(r["documents"]) > 1 else ""
        print(f"{r['ticker'] + ' FY' + str(r['fiscal_year']):<14}{pages:>7}{r['chunks']:>8}"
              f"{r['table_chunks']:>8}{r['tokens']:>10}  {secs}{extra}")
    print(f"total: {len(rows)} filings, {sum(r['chunks'] for r in rows)} chunks, "
          f"~{sum(r['tokens'] for r in rows):,} tokens")


def cmd_db_init(args) -> None:
    from finrag.index import db

    s = get_settings()
    conn = db.connect(s.database_url)
    if args.reset:
        db.reset(conn)
    db.init_schema(conn, s.embed_dim, s.embed_model)
    print(f"schema ready ({s.embed_model}, {s.embed_dim} dims)")


def cmd_index(args) -> None:
    from finrag.index.build import build_index

    s = get_settings()
    first = None
    if args.financebench_first:
        from finrag.eval import financebench

        rows = financebench.load(financebench.download(s.data_dir))
        first = financebench.question_filings(rows, load_companies())
    print(json.dumps(build_index(s, _tickers(args), args.limit, first=first), indent=2))


def cmd_ask(args) -> None:
    import time

    from finrag.index import db
    from finrag.index.embed import make_embedder
    from finrag.retrieve.vector import search

    s = get_settings()
    conn = db.connect(s.database_url)
    t0 = time.perf_counter()
    qvec = make_embedder(s).embed_query(args.question)
    t1 = time.perf_counter()
    hits = search(conn, qvec, s.embed_model, k=args.k, ticker=args.ticker and args.ticker.upper(),
                  fiscal_year=args.year, item=args.item and args.item.upper())
    t2 = time.perf_counter()
    print(f"Q: {args.question}")
    print(f"   {len(hits)} hits | embed {1000 * (t1 - t0):.0f} ms | search {1000 * (t2 - t1):.0f} ms\n")
    for n, h in enumerate(hits, 1):
        print(f"[{n}] score {h.score:.3f} | {h.citation()} | {h.chunk_type}")
        body = h.content if args.full else textwrap.shorten(h.content, 600, placeholder=" ...")
        print(textwrap.indent(body, "    "))
        print(f"    {h.source_url}\n")


def cmd_stats(args) -> None:
    from finrag.index import db

    conn = db.connect(get_settings().database_url)
    rows = conn.execute(
        """SELECT ticker, fiscal_year, count(*), count(embedding),
                  count(*) FILTER (WHERE chunk_type = 'table')
           FROM chunks GROUP BY 1, 2 ORDER BY 1, 2"""
    ).fetchall()
    print(f"{'filing':<14}{'chunks':>8}{'embedded':>10}{'tables':>8}")
    for t, fy, n, e, tab in rows:
        print(f"{t + ' FY' + str(fy):<14}{n:>8}{e:>10}{tab:>8}")
    print(f"total {sum(r[2] for r in rows)} chunks, {sum(r[3] for r in rows)} embedded")


def cmd_financebench(args) -> None:
    from finrag.eval import financebench

    s = get_settings()
    rows = financebench.load(financebench.download(s.data_dir))
    cov = financebench.coverage(rows, load_companies())
    cov.pop("in_scope_ids")
    print(json.dumps(cov, indent=2))


def cmd_page_check(args) -> None:
    from finrag.eval import financebench
    from finrag.eval.page_check import page_check
    from finrag.ingest.build import load_chunks

    s = get_settings()
    rows = financebench.load(financebench.download(s.data_dir))
    print(json.dumps(page_check(rows, load_companies(), load_chunks(s.parsed_dir)), indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="finrag", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download 10-K filings from SEC EDGAR")
    d.add_argument("--tickers", nargs="*")
    d.add_argument("--dry-run", action="store_true", help="list filings, download nothing")
    d.set_defaults(func=cmd_download)

    pa = sub.add_parser("parse", help="parse downloaded filings into chunks (JSONL)")
    pa.add_argument("--tickers", nargs="*")
    pa.set_defaults(func=cmd_parse)

    di = sub.add_parser("db-init", help="create the Postgres schema")
    di.add_argument("--reset", action="store_true", help="drop all tables first")
    di.set_defaults(func=cmd_db_init)

    ix = sub.add_parser("index", help="load chunks into Postgres and embed them")
    ix.add_argument("--tickers", nargs="*")
    ix.add_argument("--limit", type=int, help="embed at most N chunks (smoke test)")
    ix.add_argument("--financebench-first", action="store_true",
                    help="embed filings that FinanceBench questions use first")
    ix.set_defaults(func=cmd_index)

    a = sub.add_parser("ask", help="retrieve the top chunks for a question, with citations")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--ticker")
    a.add_argument("--year", type=int, help="fiscal year")
    a.add_argument("--item", help="10-K item, e.g. 7 or 8")
    a.add_argument("--full", action="store_true", help="print whole chunks")
    a.set_defaults(func=cmd_ask)

    st = sub.add_parser("stats", help="chunk and embedding counts per filing")
    st.set_defaults(func=cmd_stats)

    fb = sub.add_parser("financebench", help="FinanceBench questions covered by the corpus")
    fb.set_defaults(func=cmd_financebench)

    pc = sub.add_parser("page-check", help="compare chunk pages with FinanceBench evidence pages")
    pc.set_defaults(func=cmd_page_check)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.cmd in ("download", "index") else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        args.func(args)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
