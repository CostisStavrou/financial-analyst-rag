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
    print(json.dumps(build_index(s, _tickers(args), args.limit, first=first,
                                 reindex=args.reindex), indent=2))


def cmd_ask(args) -> None:
    import time

    from finrag.index import db
    from finrag.index.embed import make_embedder
    from finrag.retrieve import hybrid, keyword, vector

    s = get_settings()
    conn = db.connect(s.database_url)
    f = {"ticker": args.ticker and args.ticker.upper(), "fiscal_year": args.year,
         "item": args.item and args.item.upper()}
    embed = ""
    if args.method != "keyword":
        embedder = make_embedder(s)  # SDK import and client set-up kept out of the timing
        t0 = time.perf_counter()
        qvec = embedder.embed_query(args.question)
        embed = f" | embed {1000 * (time.perf_counter() - t0):.0f} ms"
    t1 = time.perf_counter()
    if args.method == "vector":
        hits = vector.search(conn, qvec, s.embed_model, k=args.k, **f)
    elif args.method == "keyword":
        hits = keyword.search(conn, args.question, k=args.k, **f)
    else:
        hits = hybrid.search(conn, args.question, qvec, s.embed_model, k=args.k, **f).hits
    t2 = time.perf_counter()
    print(f"Q: {args.question}")
    print(f"   {len(hits)} hits ({args.method}){embed} | search {1000 * (t2 - t1):.0f} ms\n")
    for n, h in enumerate(hits, 1):
        print(f"[{n}] score {h.score:.3f} | {h.citation()} | {h.chunk_type}")
        body = h.content if args.full else textwrap.shorten(h.content, 600, placeholder=" ...")
        print(textwrap.indent(body, "    "))
        print(f"    {h.source_url}\n")


def cmd_answer(args) -> None:
    from finrag.answer import generate
    from finrag.answer.llm import cost
    from finrag.index import db
    from finrag.index.embed import make_embedder
    from finrag.retrieve import vector

    s = get_settings()
    conn = db.connect(s.database_url)
    hits = vector.search(conn, make_embedder(s).embed_query(args.question), s.embed_model,
                         k=args.k, ticker=args.ticker and args.ticker.upper(),
                         fiscal_year=args.year)
    llm = generate.answerer(s)
    a, reply = generate.answer(llm, args.question, hits)
    print(f"Q: {args.question}\n")
    print(textwrap.fill(a["answer"], 100) if "\n" not in a["answer"] else a["answer"])
    if not a["answerable"]:
        print("\n(the model found no answer in the sources)")
    print("\nSources:")
    for n in a["citations"]:
        print(f"  [{n}] {hits[n - 1].citation()}\n      {hits[n - 1].source_url}")
    usd = cost(llm.model, reply.input_tokens, reply.output_tokens)
    print(f"\n{llm.model}: {reply.input_tokens} input + {reply.output_tokens} output tokens "
          f"({reply.thinking_tokens} thinking), ${usd:.4f}, {reply.ms / 1000:.1f} s"
          + (" (cached)" if reply.cached else ""))


def cmd_eval_answers(args) -> None:
    from datetime import datetime

    from finrag.answer import generate
    from finrag.answer.llm import PRICES, Budget
    from finrag.config import ROOT
    from finrag.eval import answers, financebench, retrieval
    from finrag.index import db
    from finrag.index.embed import make_embedder
    from finrag.ingest.build import load_chunks

    s = get_settings()
    rows = financebench.load(financebench.download(s.data_dir))
    refs = {r["financebench_id"]: r for r in rows}
    chunks = load_chunks(s.parsed_dir)
    questions = [q for q in retrieval.load_questions(rows, load_companies(), chunks)
                 if q.passages]
    conn = db.connect(s.database_url)
    retrieval.check_index(conn, chunks, s.embed_model)
    ctx = answers.contexts(questions, make_embedder(s), conn, s.embed_model)
    budget = Budget(args.max_usd)
    gen, judge = generate.answerer(s, budget=budget), answers.grader(s, budget=budget)
    if args.estimate:
        print(json.dumps(answers.estimate(questions, refs, ctx, gen, judge), indent=2))
        return
    results = answers.run(questions, refs, ctx, gen, judge, workers=args.workers)
    summary = answers.report(results)
    meta = {
        "questions": len(questions),
        "retrieval": f"vector search, top {answers.K} chunks (gold_pages: every chunk on the "
                     "evidence pages)",
        "generator": gen.settings(),
        "judge": judge.settings(),
        "prices_usd_per_1m": {m: PRICES[m] for m in (gen.model, judge.model)},
        "api_calls_this_run": {"generator": gen.calls, "judge": judge.calls},
        "usd_spent_this_run": round(budget.spent, 4),
        "code": _git_version(),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "PatronusAI/financebench, CC BY-NC 4.0 (question, reference and answer "
                   "text left out here)",
    }
    detail, public = retrieval.save("answers", meta, summary, results, s.data_dir,
                                    ROOT / "results", public_row=answers._public)
    print(answers.format_report(summary))
    print(f"\nAPI calls: {gen.calls} generator, {judge.calls} judge (the rest from the cache), "
          f"${budget.spent:.4f} spent")
    print(f"per-question detail: {detail}\nsummary without FinanceBench text: {public}")


def cmd_agent(args) -> None:
    import asyncio

    from finrag.agents.pipeline import Pipeline
    from finrag.index import db
    from finrag.index.embed import make_embedder

    s = get_settings()
    p = Pipeline(db.connect(s.database_url), make_embedder(s), s.embed_model, load_companies(),
                 s.gemini_api_key)
    r = asyncio.run(p.run(args.question))
    final = r["verdict"] or {}
    print(f"Q: {args.question}\n\nplan: {json.dumps(r['plan'])}\n")
    print(final.get("answer", "(no answer)"))
    if final.get("problems"):
        print("\nverifier corrected the draft: " + "; ".join(final["problems"]))
    print("\nSources:")
    for n in final.get("citations", []):
        if 1 <= n <= len(r["sources"]):
            print(f"  [{n}] {r['sources'][n - 1].citation()}")
    print()
    for step, m in r["steps"].items():
        extra = (f"{m['sources']} sources" if step == "retriever" else
                 f"{m['input_tokens']} in / {m['output_tokens']} out tokens, "
                 f"{m['tool_calls']} tool calls, ${m['usd']:.4f}")
        print(f"  {step:<10} {(m['ms'] or 0) / 1000:5.1f} s  {extra}")
    print(f"  total      {r['ms'] / 1000:5.1f} s  ${r['usd']:.4f}")


def cmd_eval_agents(args) -> None:
    from datetime import datetime

    from finrag.agents import pipeline
    from finrag.config import ROOT
    from finrag.eval import agents, answers, financebench, retrieval
    from finrag.index import db
    from finrag.index.embed import make_embedder
    from finrag.ingest.build import load_chunks

    s = get_settings()
    rows = financebench.load(financebench.download(s.data_dir))
    refs = {r["financebench_id"]: r for r in rows}
    chunks = load_chunks(s.parsed_dir)
    questions = [q for q in retrieval.load_questions(rows, load_companies(), chunks)
                 if q.passages]
    conn = db.connect(s.database_url)
    retrieval.check_index(conn, chunks, s.embed_model)
    p = pipeline.Pipeline(conn, make_embedder(s), s.embed_model, load_companies(),
                          s.gemini_api_key)
    name = f"agents_{args.tag}" if args.tag else "agents"
    runs = agents.run(questions, p, s.data_dir / "eval" / f"{name}_runs.jsonl", args.max_usd)
    results = agents.grade(questions, refs, runs, answers.grader(s))
    single = ROOT / "results" / "answers.json"
    baseline = ({r["id"]: r["unfiltered"]["label"]
                 for r in json.loads(single.read_text())["questions"]}
                if single.exists() else None)
    summary = agents.report(results, baseline)
    meta = {
        "questions": len(questions),
        "pipeline": "ADK SequentialAgent: planner -> retriever (code) -> analyst (calculator "
                    "tool) -> verifier (calculator tool)",
        "models": {"planner": pipeline.PLANNER_MODEL, "analyst": pipeline.ANALYST_MODEL,
                   "verifier": pipeline.VERIFIER_MODEL, "grader": answers.JUDGE_MODEL},
        "retrieval": {"question_top_k": pipeline.K_QUESTION, "search_top_k": pipeline.K_SEARCH,
                      "max_sources": pipeline.MAX_SOURCES, "merge": pipeline.MERGE},
        "baseline": "results/answers.json, no-filter setting",
        "code": _git_version(),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "PatronusAI/financebench, CC BY-NC 4.0 (question, reference and answer "
                   "text left out here)",
    }
    detail, public = retrieval.save(name, meta, summary, results, s.data_dir,
                                    ROOT / "results", public_row=agents.public)
    print(agents.format_report(summary))
    print(f"\nper-question detail: {detail}\nsummary without FinanceBench text: {public}")


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


def _git_version() -> str:
    import subprocess

    from finrag.config import ROOT

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout

    dirty = git("status", "--porcelain", "--", "finrag").strip()
    return git("rev-parse", "--short", "HEAD").strip() + (" + uncommitted changes" if dirty else "")


def cmd_eval_retrieval(args) -> None:
    from datetime import datetime

    from finrag.config import ROOT
    from finrag.eval import financebench, retrieval
    from finrag.index import db
    from finrag.index.embed import make_embedder
    from finrag.ingest.build import load_chunks
    from finrag.retrieve import hybrid, keyword
    from finrag.retrieve.vector import EF_SEARCH

    if (args.grid or args.timing) and args.method != "hybrid":
        raise ValueError("--grid and --timing need --method hybrid")
    s = get_settings()
    rows = financebench.load(financebench.download(s.data_dir))
    chunks = load_chunks(s.parsed_dir)
    questions = retrieval.load_questions(rows, load_companies(), chunks)
    scored = [q for q in questions if q.passages]
    if not scored:
        raise RuntimeError("No FinanceBench evidence matched the chunks. Run `finrag parse`.")
    conn = db.connect(s.database_url)
    retrieval.check_index(conn, chunks, s.embed_model)
    if args.method == "hybrid":
        retrieval.check_keyword_index(conn, chunks)
    exact = "_exact" if args.exact else ""
    if args.grid:
        grid = retrieval.fusion_grid(scored, make_embedder(s), conn, s.embed_model, k=args.k,
                                     exact=args.exact)
        meta = {"method": "hybrid fusion sensitivity check (fixed before any hybrid run)",
                "rrf_k": retrieval.GRID_RRF_K, "depth": retrieval.GRID_DEPTH, "k": args.k,
                "vector_leg": "exact" if args.exact else "HNSW where the planner uses it",
                "questions_scored": len(scored), "code": _git_version(),
                "run_at": datetime.now().isoformat(timespec="seconds")}
        name = f"retrieval_hybrid_grid{exact}" + ("" if args.k == 20 else f"_k{args.k}")
        _, public = retrieval.save(name, meta, grid, [], s.data_dir, ROOT / "results")
        print(retrieval.format_grid(grid, args.k))
        print(f"\nsummary: {public}")
        return
    if args.timing:
        import os

        load = [round(x, 1) for x in os.getloadavg()]
        timing = retrieval.time_searches(scored, make_embedder(s), conn, s.embed_model,
                                         k=args.k, passes=args.timing)
        meta = {"method": "search latency: vector leg, keyword leg, hybrid (legs one after "
                          "the other, plus fusion), each leg at hybrid depth",
                "passes": f"1 warm-up, then {args.timing} timed passes over every question",
                "depth_per_leg": hybrid.DEPTH, "k": args.k,
                "vector_leg": "HNSW where the planner uses it",
                "questions": len(scored), "code": _git_version(),
                "load_average_1_5_15min": {"before": load,
                                           "after": [round(x, 1) for x in os.getloadavg()]},
                "machine": "Postgres in Docker on the developer's Mac",
                "run_at": datetime.now().isoformat(timespec="seconds")}
        _, public = retrieval.save("latency_hybrid", meta, timing, [], s.data_dir,
                                   ROOT / "results")
        print(retrieval.format_latency(timing))
        print(f"\nsummary: {public}")
        return
    results = retrieval.run(scored, make_embedder(s), conn, s.embed_model, k=args.k,
                            exact=args.exact, method=args.method)
    summary = retrieval.report(results, args.k)
    meta = {
        "method": args.method,
        "search": "exact (no index)" if args.exact
                  else f"HNSW (ef_search {EF_SEARCH}, iterative_scan relaxed_order) where the "
                       "planner uses it, see searches_using_hnsw",
        "embedding_model": s.embed_model,
        "dims": s.embed_dim,
        "k": args.k,
        "query": "FinanceBench question, verbatim",
        "gold": {"window": retrieval.WINDOW, "step": retrieval.STEP,
                 "min_shared": "10% of windows, at least 1, at most 3",
                 "repeat_share": retrieval.REPEAT_SHARE, "min_chars": retrieval.MIN_CHARS},
        "questions_in_scope": len(questions),
        "questions_scored": len(scored),
        "excluded_no_evidence_found": [q.id for q in questions if not q.passages],
        "some_evidence_not_found": [q.id for q in questions if q.passages and q.not_found],
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "code": _git_version(),
        "dataset": "PatronusAI/financebench, CC BY-NC 4.0 (question text left out here)",
    }
    if args.method == "hybrid":
        meta["search"] = f"vector leg: {meta['search']}"
        meta["keyword"] = {
            "scorer": "BM25, Lucene idf ln(1 + (N - df + 0.5) / (df + 0.5))",
            "k1": keyword.K1, "b": keyword.B,
            "text": "Chunk.embed_text (filing/section/page header + content), the text the "
                    "embedding model sees",
            "terms": "Postgres english text search; a chunk matches on any question term",
            "query": "FinanceBench question verbatim, each term weighted by its count in it",
            "tf_dl": "lexeme positions",
            "statistics": "N, df, avgdl over the whole corpus, also when filtered",
            "ties": "score rounded to 9 decimals, then md5(id)",
            "index": {**keyword.corpus_stats(conn), "version": db.keyword_version(conn)},
        }
        meta["fusion"] = {
            "method": "reciprocal rank fusion", "rrf_k": hybrid.RRF_K,
            "depth_per_leg": hybrid.DEPTH, "weights": "equal",
            "ranks": "1-based; equal scores within a leg share a rank",
            "ties": "better vector rank, then better keyword rank, then md5(id)",
            "legs": "run one after the other; search_ms is their sum",
        }
        meta["hnsw_index_filenode"] = conn.execute(
            "SELECT pg_relation_filenode('chunks_embedding_idx')").fetchone()[0]
        meta["vector_leg_vs_baseline"] = retrieval.same_as_baseline(
            results, ROOT / "results" / f"retrieval_vector{exact}.json")
    # the default k=20 runs are the ones the README cites; other k get their own files
    name = (f"retrieval_{args.method}{exact}" + ("" if args.k == 20 else f"_k{args.k}"))
    detail, public = retrieval.save(name, meta, summary, results, s.data_dir, ROOT / "results")
    print(f"{meta['method']} search, {meta['search']}: {len(scored)} of {len(questions)} "
          f"in-scope questions scored, top {args.k}\n")
    print(retrieval.format_report(summary, args.k))
    if args.method == "hybrid" and (same := meta["vector_leg_vs_baseline"]):
        print(f"\nvector leg vs {same['file']}: {same['identical']}/{same['searches']} top-"
              f"{same['top']} lists identical")
    print(f"\nper-question detail: {detail}\nsummary without question text: {public}")


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
    ix.add_argument("--reindex", action="store_true",
                    help="rebuild the HNSW index even if no new vectors were written")
    ix.set_defaults(func=cmd_index)

    a = sub.add_parser("ask", help="retrieve the top chunks for a question, with citations")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--ticker")
    a.add_argument("--year", type=int, help="fiscal year")
    a.add_argument("--item", help="10-K item, e.g. 7 or 8")
    a.add_argument("--full", action="store_true", help="print whole chunks")
    a.add_argument("--method", choices=("vector", "keyword", "hybrid"), default="vector")
    a.set_defaults(func=cmd_ask)

    an = sub.add_parser("answer", help="answer a question with Gemini from the top-k chunks")
    an.add_argument("question")
    an.add_argument("-k", type=int, default=10)
    an.add_argument("--ticker")
    an.add_argument("--year", type=int, help="fiscal year")
    an.set_defaults(func=cmd_answer)

    ea = sub.add_parser("eval-answers",
                        help="FinanceBench answer accuracy, graded by a second Gemini model")
    ea.add_argument("--estimate", action="store_true",
                    help="count input tokens and estimate the cost; no generation calls "
                         "(uncached query embeddings are still made)")
    ea.add_argument("--workers", type=int, default=4, help="API calls at a time")
    ea.add_argument("--max-usd", type=float, help="stop once fresh calls have cost this much")
    ea.set_defaults(func=cmd_eval_answers)

    ag = sub.add_parser("agent", help="answer a question with the multi-agent pipeline (ADK)")
    ag.add_argument("question")
    ag.set_defaults(func=cmd_agent)

    ev = sub.add_parser("eval-agents", help="FinanceBench answer accuracy of the agent pipeline")
    ev.add_argument("--max-usd", type=float, help="stop once the agent runs have cost this much")
    ev.add_argument("--tag", help="name a new run: results/agents_<tag>.json, its own runs file")
    ev.set_defaults(func=cmd_eval_agents)

    st = sub.add_parser("stats", help="chunk and embedding counts per filing")
    st.set_defaults(func=cmd_stats)

    fb = sub.add_parser("financebench", help="FinanceBench questions covered by the corpus")
    fb.set_defaults(func=cmd_financebench)

    pc = sub.add_parser("page-check", help="compare chunk pages with FinanceBench evidence pages")
    pc.set_defaults(func=cmd_page_check)

    er = sub.add_parser("eval-retrieval",
                        help="FinanceBench retrieval score: is the evidence in the top k?")
    er.add_argument("-k", type=int, default=20, choices=(1, 3, 5, 10, 20))
    er.add_argument("--exact", action="store_true",
                    help="exact (brute-force) vector search instead of the HNSW index")
    er.add_argument("--method", choices=("vector", "hybrid"), default="vector",
                    help="hybrid also scores its vector and keyword legs")
    er.add_argument("--grid", action="store_true",
                    help="hybrid only: fusion sensitivity check over RRF k and depth")
    er.add_argument("--timing", type=int, metavar="PASSES",
                    help="hybrid only: time each search over PASSES passes after a warm-up")
    er.set_defaults(func=cmd_eval_retrieval)

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
