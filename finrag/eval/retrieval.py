"""Retrieval evaluation on FinanceBench: is the evidence in the top k? No LLM involved.

Gold labels come from each question's evidence passages (`evidence_text`, copied from the
company's PDF). Text is compared in lowercase with only letters and digits kept, as
60-character windows every 30 characters. Within the question's filing:

- core chunks share the most windows with the passage, or repeat at least half of it
  (the same sentence in MD&A and in the notes, say);
- other chunks on a core chunk's pages are gold too if they share 10% of the windows
  (at least 1, at most 3): a table split over two chunks is all gold, a column header
  shared with a table on another page is not.

Passages under 40 characters after normalising are too short to place and count as not found;
a question's passages that are not found count as never retrieved.

Gold labels only see word-for-word copies: a reworded passage, or the same figures in a table
with a different header, is not gold even when it answers the question.

Gold pages are (ticker, fiscal year, document, page) of the gold chunks: CVS FY2018 has a
page 40 in both its 10-K and its Exhibit 13. A chunk spanning two pages counts both.

Metrics per question, over the top k chunks:
- page hit@k: a retrieved chunk's pages overlap a gold page (the primary metric);
- chunk hit@k: a gold chunk itself is retrieved;
- MRR: 1 / rank of the first page hit, 0 if none in the top k;
- full recall@k, for questions with several passages: every passage has a page hit.

Each question runs three ways: unfiltered (all filings), filtered to its company, and
filtered to its company and fiscal year (the right filing, as an oracle would pick).

Methods: `vector` alone, or `hybrid`, which also scores its two legs (vector and keyword,
each cut to the top k) from the same searches, so all three rows come from one run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import psycopg

from finrag.config import Company
from finrag.eval.financebench import in_scope
from finrag.index.db import KEYWORD_VERSION, keyword_version
from finrag.index.embed import Embedder
from finrag.ingest.chunk import Chunk
from finrag.retrieve import hybrid, keyword
from finrag.retrieve.vector import Hit, search, uses_hnsw

WINDOW, STEP = 60, 30
REPEAT_SHARE = 0.5  # share of a passage a chunk elsewhere must repeat to be gold
MIN_CHARS = 40  # shorter passages ("Net income") recur all over a filing
KS = (1, 3, 5, 10, 20)
SETTINGS = ("unfiltered", "company", "company_year")
LEGS = ("vector", "keyword")
# Fusion sensitivity check, fixed before the first hybrid run (see the README)
GRID_RRF_K = (10, 30, 60, 100)
GRID_DEPTH = (20, 50, 100)

Page = tuple[str, int, str, int]  # ticker, fiscal year, document, page

_NORM = re.compile(r"[^a-z0-9]+")


def normalise(text: str) -> str:
    return _NORM.sub("", text.lower())


def windows(text: str) -> list[str]:
    """WINDOW-character windows, STEP apart; the last one ends at the end of the text."""
    if len(text) <= WINDOW:
        return [text] if text else []
    starts = list(range(0, len(text) - WINDOW + 1, STEP))
    if starts[-1] != len(text) - WINDOW:
        starts.append(len(text) - WINDOW)
    return [text[s : s + WINDOW] for s in starts]


def min_shared(n_windows: int) -> int:
    """Windows a chunk must share with a passage: 10% of them, at least 1, at most 3."""
    return max(1, min(3, math.ceil(0.1 * n_windows)))


def pages_of(c: Chunk | Hit) -> set[Page]:
    return {(c.ticker, c.fiscal_year, c.document, p) for p in range(c.page_start, c.page_end + 1)}


@dataclass
class Passage:
    chunk_ids: set[str]
    pages: set[Page]


@dataclass
class Question:
    id: str
    ticker: str
    fiscal_year: int
    question: str
    question_type: str
    passages: list[Passage]  # evidence passages found in our chunks
    not_found: int  # evidence passages with no matching chunk

    @property
    def chunk_ids(self) -> set[str]:
        return set().union(*(p.chunk_ids for p in self.passages))

    @property
    def pages(self) -> set[Page]:
        return set().union(*(p.pages for p in self.passages))


def gold_passage(evidence: str, chunks: list[tuple[str, Chunk]]) -> Passage | None:
    """Gold chunks for one evidence passage. `chunks`: (normalised content, chunk) for every
    chunk of the question's filing. None if the passage is too short to place or no chunk
    shares enough of it."""
    text = normalise(evidence)
    if len(text) < MIN_CHARS:
        return None
    ws = windows(text)
    shared = [(sum(w in content for w in ws), c) for content, c in chunks]
    best = max((n for n, _ in shared), default=0)
    need = min_shared(len(ws))
    if best < need:
        return None
    repeat = max(need, math.ceil(REPEAT_SHARE * len(ws)))
    core_pages = set().union(*(pages_of(c) for n, c in shared if n == best or n >= repeat))
    gold = [c for n, c in shared if n >= need and pages_of(c) & core_pages]
    return Passage({c.id for c in gold}, set().union(*(pages_of(c) for c in gold)))


def load_questions(rows: list[dict], companies: list[Company], chunks: list[Chunk]
                   ) -> list[Question]:
    by_filing: dict[tuple[str, int], list[tuple[str, Chunk]]] = defaultdict(list)
    for c in chunks:
        by_filing[(c.ticker, c.fiscal_year)].append((normalise(c.content), c))
    questions = []
    for r, co in in_scope(rows, companies):
        fy = int(r["doc_period"])
        evidence = r["evidence"]
        if isinstance(evidence, str):
            evidence = ast.literal_eval(evidence)
        found = [gold_passage(e["evidence_text"], by_filing[(co.ticker, fy)]) for e in evidence]
        questions.append(Question(
            id=r["financebench_id"], ticker=co.ticker, fiscal_year=fy, question=r["question"],
            question_type=r["question_type"], passages=[p for p in found if p],
            not_found=sum(p is None for p in found),
        ))
    return questions


def check_index(conn: psycopg.Connection, chunks: list[Chunk], model: str) -> None:
    """Gold labels come from the chunk files, hits from the database, so both must hold the
    same chunks: ids are numbered by position, and a re-parse shifts them. Compares the content
    hash and every field a hit is scored on (its filing, document and pages)."""
    in_db = {r[0]: (tuple(r[1:7]), r[7], r[8]) for r in conn.execute(
        "SELECT id, content_sha, ticker, fiscal_year, document, page_start, page_end, "
        "accession, embedding IS NOT NULL AND embedding_model = %s FROM chunks", (model,))}
    in_files = {c.id: (c.content_sha, c.ticker, c.fiscal_year, c.document, c.page_start,
                       c.page_end) for c in chunks}
    accessions = {c.accession for c in chunks}
    only_db = in_db.keys() - in_files.keys()
    # `index` only prunes filings that still have a chunk file
    orphans = {i for i in only_db if in_db[i][1] not in accessions}
    both = in_files.keys() & in_db.keys()
    problems = {
        "missing from the database": in_files.keys() - in_db.keys(),
        "only in the database": only_db - orphans,
        "changed since they were indexed": {i for i in both if in_files[i] != in_db[i][0]},
        f"without a {model} embedding": {i for i, (_, _, ok) in in_db.items() if not ok},
    }
    found = "; ".join(f"{len(ids)} chunks {what} (first: {min(ids)})"
                      for what, ids in problems.items() if ids)
    msgs = [f"{found}. Run `finrag index`."] if found else []
    if orphans:
        msgs.append(f"{len(orphans)} chunks of filings with no chunk file (first: "
                    f"{min(orphans)}). Rebuild with `finrag db-init --reset` and "
                    "`finrag index` (embeddings come from the cache).")
    if msgs:
        raise RuntimeError(f"The database doesn't match the chunk files: {' '.join(msgs)}")


def check_keyword_index(conn: psycopg.Connection, chunks: list[Chunk]) -> None:
    """The keyword index must be the version this code defines and hold, for exactly the
    chunks in the files, the text the embedding model sees (Chunk.embed_text). `finrag index`
    rebuilds it."""
    try:
        in_db = {r[0]: r[1] for r in conn.execute("SELECT id, text_md5 FROM keyword_docs", ())}
    except keyword.MISSING as e:
        raise RuntimeError(keyword.NOT_BUILT) from e
    if (version := keyword_version(conn)) != KEYWORD_VERSION:
        raise RuntimeError(f"The keyword index is version {version}, this code expects "
                           f"{KEYWORD_VERSION}. Run `finrag index`.")
    want = {c.id: hashlib.md5(c.embed_text().encode()).hexdigest() for c in chunks}
    problems = {
        "missing from": want.keys() - in_db.keys(),
        "only in": in_db.keys() - want.keys(),
        "with other text in": {i for i in want.keys() & in_db.keys() if want[i] != in_db[i]},
    }
    found = "; ".join(f"{len(ids)} chunks {what} the keyword index (first: {min(ids)})"
                      for what, ids in problems.items() if ids)
    if found:
        raise RuntimeError(f"{found}. Run `finrag index`.")


def _first(hits: list[Hit], match) -> int | None:
    """1-based rank of the first matching hit, None if there is none."""
    return next((i for i, h in enumerate(hits, 1) if match(h)), None)


def score(q: Question, hits: list[Hit]) -> dict:
    """Rank of the first page hit, chunk hit and page hit per passage (None: not retrieved)."""
    pages, ids = q.pages, q.chunk_ids
    per_passage = [_first(hits, lambda h, p=p: pages_of(h) & p.pages) for p in q.passages]
    per_passage += [None] * q.not_found  # a passage we couldn't place is never retrieved
    return {
        "page_rank": _first(hits, lambda h: pages_of(h) & pages),
        "chunk_rank": _first(hits, lambda h: h.id in ids),
        "passage_ranks": per_passage,
        # rank by which every passage has a page hit
        "all_rank": None if None in per_passage else max(per_passage, default=None),
    }


def summarise(scores: list[dict], k: int) -> dict:
    """Hit rates, MRR and full recall over a list of `score` results."""
    ks = [x for x in KS if x <= k]

    def share(rows: list[dict], key: str, cut: int) -> float | None:
        if not rows:
            return None
        return round(sum(r[key] is not None and r[key] <= cut for r in rows) / len(rows), 4)

    n = len(scores)
    if not n:
        return {"questions": 0}
    multi = [s for s in scores if len(s["passage_ranks"]) > 1]
    out: dict = {"questions": n}
    out |= {f"page_hit@{x}": share(scores, "page_rank", x) for x in ks}
    out |= {f"chunk_hit@{x}": share(scores, "chunk_rank", x) for x in ks}
    out["page_mrr"] = round(sum(1 / s["page_rank"] for s in scores if s["page_rank"]) / n, 4)
    out["chunk_mrr"] = round(sum(1 / s["chunk_rank"] for s in scores if s["chunk_rank"]) / n, 4)
    out["multi_passage_questions"] = len(multi)
    out |= {f"full_recall@{x}": share(multi, "all_rank", x) for x in ks}
    return out


def _filters(q: Question) -> dict[str, dict]:
    return {
        "unfiltered": {},
        "company": {"ticker": q.ticker},
        "company_year": {"ticker": q.ticker, "fiscal_year": q.fiscal_year},
    }


def _hybrid_result(q: Question, res: hybrid.HybridResult, k: int, used_hnsw: bool) -> dict:
    """Scores of the fused top k and of each leg's top k, plus where the fused hits came from."""
    legs = {"vector": res.vector, "keyword": res.keyword}
    return {
        **score(q, res.hits),
        "search_ms": round(sum(res.ms.values()), 1),
        "used_hnsw": used_hnsw,  # by the vector leg, at depth DEPTH
        "overlap": len({h.id for h in res.vector} & {h.id for h in res.keyword}),
        "retrieved": [[h.id, h.page_start, h.page_end, round(h.score, 6), r["vector"],
                       r["keyword"]] for h, r in zip(res.hits, res.ranks)],
        "legs": {leg: {
            **score(q, hits[:k]),
            "page_rank_depth": score(q, hits)["page_rank"],  # within the leg's full list
            "search_ms": round(res.ms[leg], 1),
            "retrieved": [h.id for h in hits],
        } for leg, hits in legs.items()},
    }


def run(questions: list[Question], embedder: Embedder, conn: psycopg.Connection, model: str,
        k: int = 20, exact: bool = False, method: str = "vector") -> list[dict]:
    """Retrieve the top k for each question in each setting. One result row per question."""
    rows = []
    for q in questions:
        cache_hits = embedder.cache_hits
        t0 = time.perf_counter()
        qvec = embedder.embed_query(q.question)
        row = {
            "id": q.id,
            "ticker": q.ticker,
            "fiscal_year": q.fiscal_year,
            "question_type": q.question_type,
            "question": q.question,
            "passages": len(q.passages),
            "passages_not_found": q.not_found,
            "gold_chunks": sorted(q.chunk_ids),
            "gold_pages": sorted(q.pages),
            "embed_ms": round(1000 * (time.perf_counter() - t0), 1),
            "embed_cached": embedder.cache_hits > cache_hits,
        }
        for setting, f in _filters(q).items():
            if method == "hybrid":
                res = hybrid.search(conn, q.question, qvec, model, k=k, exact=exact, **f)
                used = not exact and uses_hnsw(conn, qvec, model, k=hybrid.DEPTH, **f)
                row[setting] = _hybrid_result(q, res, k, used)
                continue
            t0 = time.perf_counter()
            hits = search(conn, qvec, model, k=k, exact=exact, **f)
            row[setting] = {
                **score(q, hits),
                "search_ms": round(1000 * (time.perf_counter() - t0), 1),
                # with a selective filter Postgres may skip HNSW and sort exactly
                "used_hnsw": not exact and uses_hnsw(conn, qvec, model, k=k, **f),
                "retrieved": [[h.id, h.page_start, h.page_end, round(h.score, 4)] for h in hits],
            }
        rows.append(row)
    return rows


def _p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def wilson(x: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for x hits out of n."""
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar p-value: the questions only one method gets right, against a
    fair coin."""
    n = fixed + broken
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _hit(s: dict, cut: int) -> bool:
    return s["page_rank"] is not None and s["page_rank"] <= cut


def paired(rows: list[dict], setting: str, k: int) -> dict:
    """Hybrid against its own vector leg, per cut-off: questions it fixes (hybrid hits, vector
    doesn't) and breaks (the reverse), with an exact McNemar p-value."""
    out = {}
    for x in (c for c in KS if c <= k):
        fixed = [r["id"] for r in rows
                 if _hit(r[setting], x) and not _hit(r[setting]["legs"]["vector"], x)]
        broken = [r["id"] for r in rows
                  if _hit(r[setting]["legs"]["vector"], x) and not _hit(r[setting], x)]
        out[f"page_hit@{x}"] = {"fixed": fixed, "broken": broken,
                                "mcnemar_p": round(mcnemar_exact(len(fixed), len(broken)), 4)}
    return out


def _metrics(rows: list[dict], get, k: int) -> dict:
    """summarise() of get(row) over rows, with latency and breakdowns by type and company."""
    by_type, by_company = defaultdict(list), defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(get(r))
        by_company[r["ticker"]].append(get(r))
    ms = [get(r)["search_ms"] for r in rows]
    return {
        **summarise([get(r) for r in rows], k),
        "search_ms_median": round(statistics.median(ms), 1),
        "search_ms_p95": round(_p95(ms), 1),
        "by_question_type": {t: summarise(v, k) for t, v in sorted(by_type.items())},
        "by_company": {t: summarise(v, k) for t, v in sorted(by_company.items())},
    }


def report(rows: list[dict], k: int) -> dict:
    """Metrics per setting, overall and by question type and company, plus latency. For
    hybrid runs, the same for each leg, and hybrid against its vector leg."""
    out: dict = {}
    h = min(5, k)
    for setting in SETTINGS:
        m = _metrics(rows, lambda r, s=setting: r[s], k)
        m["searches_using_hnsw"] = sum(r[setting]["used_hnsw"] for r in rows)
        if "legs" in rows[0][setting]:
            m["legs"] = {
                leg: _metrics(rows, lambda r, s=setting, leg=leg: r[s]["legs"][leg], k)
                for leg in LEGS}
            m["legs_overlap_median"] = statistics.median(r[setting]["overlap"] for r in rows)
            m["vs_vector"] = paired(rows, setting, k)
            n = len(rows)
            m[f"page_hit@{h}_wilson95"] = {
                name: [round(x, 4) for x in wilson(round(mm[f"page_hit@{h}"] * n), n)]
                for name, mm in (("vector", m["legs"]["vector"]),
                                 ("keyword", m["legs"]["keyword"]), ("hybrid", m))
            }
        out[setting] = m
    fresh = [r["embed_ms"] for r in rows if not r["embed_cached"]]
    out["query_embedding"] = {
        "api_calls": len(fresh),
        "from_cache": len(rows) - len(fresh),
        "ms_median": round(statistics.median(fresh), 1) if fresh else None,
        "ms_p95": round(_p95(fresh), 1) if fresh else None,
    }
    return out


def same_as_baseline(rows: list[dict], path: Path) -> dict | None:
    """Whether a hybrid run's vector leg returned the same top chunks, in the same order, as a
    committed vector run (its top k, usually 20), so hybrid and the baseline differ only by
    the fusion."""
    if not path.exists():
        return None
    base = {r["id"]: r for r in json.loads(path.read_text())["questions"]}
    differ, top = [], 0
    for r in rows:
        for s in SETTINGS:
            want = base[r["id"]][s]["retrieved"] if r["id"] in base else None
            got = r[s]["legs"]["vector"]["retrieved"]
            top = max(top, len(want or []))
            if want is None or got[: len(want)] != want:
                differ.append(f"{r['id']} {s}")
    return {"file": path.name, "top": top, "searches": len(rows) * len(SETTINGS),
            "identical": len(rows) * len(SETTINGS) - len(differ), "differ": differ}


def fusion_grid(questions: list[Question], embedder: Embedder, conn: psycopg.Connection,
                model: str, k: int = 20, exact: bool = False) -> dict:
    """Sensitivity check, fixed in advance and not used to choose the settings: RRF k by
    depth, plus the chosen cell with keyword-first tie-breaking, each against the vector leg
    alone (questions fixed and broken at hit@5). Each leg is searched once, at the largest
    depth, and cut; a cut list can differ slightly from a search at that depth (the planner
    may pick another plan), so the chosen cell is a cross-check of the main run."""
    deepest = max(GRID_DEPTH)
    cells: dict = defaultdict(lambda: defaultdict(list))  # setting -> cell -> [(id, score)]
    for q in questions:
        qvec = embedder.embed_query(q.question)
        for setting, f in _filters(q).items():
            legs = {"vector": search(conn, qvec, model, k=deepest, exact=exact, **f),
                    "keyword": keyword.search(conn, q.question, k=deepest, **f)}
            by_id = {h.id: h for hits in legs.values() for h in hits}
            cells[setting]["vector_only"].append((q.id, score(q, legs["vector"][:k])))
            runs = [(f"k{rk}_depth{d}", rk, d, LEGS) for rk in GRID_RRF_K for d in GRID_DEPTH]
            runs.append((f"k{hybrid.RRF_K}_depth{hybrid.DEPTH}_keyword_first", hybrid.RRF_K,
                         hybrid.DEPTH, LEGS[::-1]))
            for name, rk, d, order in runs:
                fused = hybrid.fuse({leg: legs[leg][:d] for leg in order}, rk)[:k]
                cells[setting][name].append((q.id, score(q, [by_id[x.id] for x in fused])))
    h = min(5, k)
    out: dict = {}
    for setting, c in cells.items():
        base = [sc for _, sc in c["vector_only"]]
        out[setting] = {}
        for name, scored in c.items():
            m = summarise([sc for _, sc in scored], k)
            if name != "vector_only":
                pairs = [(qid, _hit(sc, h), _hit(b, h)) for (qid, sc), b in zip(scored, base)]
                fixed = [qid for qid, a, b in pairs if a and not b]
                broken = [qid for qid, a, b in pairs if b and not a]
                m[f"vs_vector@{h}"] = {"fixed": fixed, "broken": broken, "mcnemar_p":
                                       round(mcnemar_exact(len(fixed), len(broken)), 4)}
            out[setting][name] = m
    return out


def time_searches(questions: list[Question], embedder: Embedder, conn: psycopg.Connection,
                  model: str, k: int = 20, passes: int = 3) -> dict:
    """Search latency, median and p95 per method and setting: one warm-up pass, then
    `passes` timed passes over every question. Each leg is timed at hybrid depth, then the
    whole hybrid search (legs one after the other, plus fusion)."""
    vecs = [embedder.embed_query(q.question) for q in questions]
    ms: dict = {s: {m: [] for m in ("vector", "keyword", "hybrid")} for s in SETTINGS}
    for n in range(passes + 1):
        for q, v in zip(questions, vecs):
            for setting, f in _filters(q).items():
                t0 = time.perf_counter()
                search(conn, v, model, k=hybrid.DEPTH, **f)
                t1 = time.perf_counter()
                keyword.search(conn, q.question, k=hybrid.DEPTH, **f)
                t2 = time.perf_counter()
                hybrid.search(conn, q.question, v, model, k=k, **f)
                t3 = time.perf_counter()
                if n:  # pass 0 warms the caches
                    for m, t in (("vector", t1 - t0), ("keyword", t2 - t1), ("hybrid", t3 - t2)):
                        ms[setting][m].append(1000 * t)
    return {s: {m: {"ms_median": round(statistics.median(v), 1), "ms_p95": round(_p95(v), 1),
                    "samples": len(v)} for m, v in by_method.items()}
            for s, by_method in ms.items()}


def _public(row: dict) -> dict:
    """A result row without question text, and retrieved chunks as ids only (with their rank
    in each leg, for hybrid runs)."""
    out = {k: v for k, v in row.items() if k != "question"}
    for s in SETTINGS:
        r = row[s]
        out[s] = {**r, "retrieved": [x[0] if len(x) == 4 else [x[0], x[4], x[5]]
                                     for x in r["retrieved"]]}
        if "legs" in r:
            out[s]["legs"] = {leg: {key: v for key, v in m.items() if key != "retrieved"}
                              for leg, m in r["legs"].items()}
    return out


def save(name: str, meta: dict, summary: dict, rows: list[dict], data_dir: Path,
         results_dir: Path) -> tuple[Path, Path]:
    """Full detail to data/eval/ (git-ignored). A copy without question text goes to
    results/, since FinanceBench is CC BY-NC 4.0 and is not redistributed here.
    One line per question, so runs diff cleanly."""
    detail = data_dir / "eval" / f"{name}.json"
    public = results_dir / f"{name}.json"
    for path, out_rows in ((detail, rows), (public, [_public(r) for r in rows])):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ",\n".join(json.dumps(r) for r in out_rows)
        path.write_text(f'{{"meta": {json.dumps(meta, indent=1)},\n'
                        f'"summary": {json.dumps(summary, indent=1)},\n'
                        f'"questions": [\n{lines}\n]}}\n')
    return detail, public


def _pct(x: float | None, width: int) -> str:
    s = "–" if x is None else f"{x:.0%}"
    return f"{s:>{width}}"


def _row(label: str, m: dict, k: int, hnsw: str) -> str:
    ks, h = [x for x in KS if x <= k], min(5, k)
    return (f"{label:<14}" + "".join(_pct(m[f"page_hit@{x}"], 6) for x in ks)
            + f"{m['page_mrr']:>7.2f}" + _pct(m[f"chunk_hit@{h}"], 9)
            + _pct(m[f"full_recall@{h}"], 7) + _pct(m[f"full_recall@{k}"], 7)
            + f"{m['search_ms_median']:>11.0f}{m['search_ms_p95']:>6.0f}{hnsw:>7}")


def format_report(summary: dict, k: int) -> str:
    ks, h = [x for x in KS if x <= k], min(5, k)
    lines = [
        f"{'page hit@k':<14}" + "".join(f"{'@' + str(x):>6}" for x in ks)
        + f"{'MRR':>7}{'chunk@' + str(h):>9}{'all@' + str(h):>7}{'all@' + str(k):>7}"
        + f"{'search ms':>11}{'p95':>6}{'HNSW':>7}"
    ]
    is_hybrid = "legs" in summary["unfiltered"]
    for setting in SETTINGS:
        m = summary[setting]
        hnsw = f"{m['searches_using_hnsw']}/{m['questions']}"
        if not is_hybrid:
            lines.append(_row(setting, m, k, hnsw))
            continue
        lines.append(setting)
        lines.append(_row("  vector", m["legs"]["vector"], k, hnsw))
        lines.append(_row("  keyword", m["legs"]["keyword"], k, "–"))
        lines.append(_row("  hybrid", m, k, hnsw))
    multi = summary["unfiltered"]["multi_passage_questions"]
    lines.append(f"(all@k: every evidence passage found, {multi} multi-passage questions; "
                 "HNSW: searches the planner ran on the HNSW index, the rest sorted exactly)")
    if is_hybrid:
        lines.append("\nhybrid vs its vector leg: questions fixed/broken, exact McNemar p")
        lines.append(f"{'':<14}" + "".join(f"{'@' + str(x):>14}" for x in ks))
        for setting in SETTINGS:
            cells = "".join(
                f"{len(c['fixed'])}/{len(c['broken'])} p{c['mcnemar_p']:.2f}".rjust(14)
                for c in (summary[setting]["vs_vector"][f"page_hit@{x}"] for x in ks))
            lines.append(f"{setting:<14}{cells}")
        u = summary["unfiltered"]
        cis = ", ".join(f"{name} {lo:.0%}–{hi:.0%}"
                        for name, (lo, hi) in u[f"page_hit@{h}_wilson95"].items())
        lines.append(f"page hit@{h} no filter, 95% CI: {cis}")
        lines.append("legs' top-50 overlap, median: " + ", ".join(
            f"{s} {summary[s]['legs_overlap_median']:g}" for s in SETTINGS))
        groups = [("vector", u["legs"]["vector"]), ("keyword", u["legs"]["keyword"]),
                  ("hybrid", u)]
        lines.append(f"\npage hit@{h} by group, no filter" + "".join(
            f"{name:>14}" for name, _ in groups))
    else:
        groups = [(s, summary[s]) for s in SETTINGS]
        lines.append(f"\npage hit@{h} by group" + "".join(f"{s:>14}" for s, _ in groups))
    for group in ("by_question_type", "by_company"):
        for name, m in summary["unfiltered"][group].items():
            label = f"{name} ({m['questions']})"
            cells = "".join(_pct(g[group][name][f"page_hit@{h}"], 14) for _, g in groups)
            lines.append(f"  {label:<22}{cells}")
    q = summary["query_embedding"]
    lines.append(f"\nquery embedding: {q['api_calls']} API calls (median {q['ms_median']} ms, "
                 f"p95 {q['ms_p95']} ms), {q['from_cache']} from cache")
    return "\n".join(lines)


def format_grid(grid: dict, k: int) -> str:
    ks, h = sorted({1, min(5, k), k}), min(5, k)
    chosen = f"k{hybrid.RRF_K}_depth{hybrid.DEPTH}"
    lines = []
    for setting, cells in grid.items():
        lines.append(f"{setting:<34}" + "".join(f"{'@' + str(x):>6}" for x in ks)
                     + f"{'MRR':>7}{'vs vector @' + str(h):>18}")
        for name, m in cells.items():
            mark = " *" if name == chosen else ""
            vs = m.get(f"vs_vector@{h}")
            cell = f"{len(vs['fixed'])}/{len(vs['broken'])} p{vs['mcnemar_p']:.2f}" if vs else ""
            lines.append(f"  {name + mark:<32}" + "".join(_pct(m[f"page_hit@{x}"], 6) for x in ks)
                         + f"{m['page_mrr']:>7.2f}{cell:>18}")
    lines.append("(* the settings used for the hybrid results, fixed before any run; "
                 "vs vector: questions fixed/broken, exact McNemar p)")
    return "\n".join(lines)


def format_latency(timing: dict) -> str:
    lines = [f"{'search ms':<14}" + "".join(f"{m + ' med/p95':>22}"
                                             for m in ("vector", "keyword", "hybrid"))]
    for setting, by_method in timing.items():
        lines.append(f"{setting:<14}" + "".join(
            f"{x['ms_median']:>13.1f} / {x['ms_p95']:<6.1f}" for x in by_method.values()))
    return "\n".join(lines)
