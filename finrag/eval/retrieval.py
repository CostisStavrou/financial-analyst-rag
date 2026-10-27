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
"""

from __future__ import annotations

import ast
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
from finrag.index.embed import Embedder
from finrag.ingest.chunk import Chunk
from finrag.retrieve.vector import Hit, search, uses_hnsw

WINDOW, STEP = 60, 30
REPEAT_SHARE = 0.5  # share of a passage a chunk elsewhere must repeat to be gold
MIN_CHARS = 40  # shorter passages ("Net income") recur all over a filing
KS = (1, 3, 5, 10, 20)
SETTINGS = ("unfiltered", "company", "company_year")

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


def run(questions: list[Question], embedder: Embedder, conn: psycopg.Connection, model: str,
        k: int = 20, exact: bool = False) -> list[dict]:
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
        filters = {
            "unfiltered": {},
            "company": {"ticker": q.ticker},
            "company_year": {"ticker": q.ticker, "fiscal_year": q.fiscal_year},
        }
        for setting, f in filters.items():
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


def report(rows: list[dict], k: int) -> dict:
    """Metrics per setting, overall and by question type and company, plus latency."""
    out: dict = {}
    for setting in SETTINGS:
        by_type, by_company = defaultdict(list), defaultdict(list)
        for r in rows:
            by_type[r["question_type"]].append(r[setting])
            by_company[r["ticker"]].append(r[setting])
        ms = [r[setting]["search_ms"] for r in rows]
        out[setting] = {
            **summarise([r[setting] for r in rows], k),
            "searches_using_hnsw": sum(r[setting]["used_hnsw"] for r in rows),
            "search_ms_median": round(statistics.median(ms), 1),
            "search_ms_p95": round(_p95(ms), 1),
            "by_question_type": {t: summarise(v, k) for t, v in sorted(by_type.items())},
            "by_company": {t: summarise(v, k) for t, v in sorted(by_company.items())},
        }
    fresh = [r["embed_ms"] for r in rows if not r["embed_cached"]]
    out["query_embedding"] = {
        "api_calls": len(fresh),
        "from_cache": len(rows) - len(fresh),
        "ms_median": round(statistics.median(fresh), 1) if fresh else None,
        "ms_p95": round(_p95(fresh), 1) if fresh else None,
    }
    return out


def _public(row: dict) -> dict:
    """A result row without question text, and retrieved chunks as ids only."""
    out = {k: v for k, v in row.items() if k != "question"}
    for s in SETTINGS:
        out[s] = {**row[s], "retrieved": [r[0] for r in row[s]["retrieved"]]}
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


def format_report(summary: dict, k: int) -> str:
    ks = [x for x in KS if x <= k]
    h = min(5, k)
    lines = [
        f"{'page hit@k':<14}" + "".join(f"{'@' + str(x):>6}" for x in ks)
        + f"{'MRR':>7}{'chunk@' + str(h):>9}{'all@' + str(h):>7}{'all@' + str(k):>7}"
        + f"{'search ms':>11}{'p95':>6}{'HNSW':>7}"
    ]
    for setting in SETTINGS:
        m = summary[setting]
        lines.append(
            f"{setting:<14}" + "".join(_pct(m[f"page_hit@{x}"], 6) for x in ks)
            + f"{m['page_mrr']:>7.2f}" + _pct(m[f"chunk_hit@{h}"], 9)
            + _pct(m[f"full_recall@{h}"], 7) + _pct(m[f"full_recall@{k}"], 7)
            + f"{m['search_ms_median']:>11.0f}{m['search_ms_p95']:>6.0f}"
            + f"{m['searches_using_hnsw']:>4}/{m['questions']}"
        )
    multi = summary["unfiltered"]["multi_passage_questions"]
    lines.append(f"(all@k: every evidence passage found, {multi} multi-passage questions; "
                 "HNSW: searches the planner ran on the HNSW index, the rest sorted exactly)")
    lines.append(f"\npage hit@{h} by group" + "".join(f"{s:>14}" for s in SETTINGS))
    for group in ("by_question_type", "by_company"):
        for name, m in summary["unfiltered"][group].items():
            label = f"{name} ({m['questions']})"
            cells = "".join(_pct(summary[s][group][name][f"page_hit@{h}"], 14)
                            for s in SETTINGS)
            lines.append(f"  {label:<22}{cells}")
    q = summary["query_embedding"]
    lines.append(f"\nquery embedding: {q['api_calls']} API calls (median {q['ms_median']} ms, "
                 f"p95 {q['ms_p95']} ms), {q['from_cache']} from cache")
    return "\n".join(lines)
