"""Answer accuracy on FinanceBench: retrieve, answer with Gemini, grade against the reference.

Each in-scope question is answered three ways, by the same model and prompt:
- unfiltered: the top K chunks by vector search over all 30 filings (the system as built);
- company_year: the top K within the question's filing (company and year known);
- gold_pages: every chunk on the FinanceBench evidence pages (retrieval taken out, so what
  is left is reading and arithmetic).

A second, larger Gemini model grades each answer against FinanceBench's reference answer:
correct, incorrect or refusal. The grade is checked against an independent second grader
(see the README).
"""

from __future__ import annotations

import math
import statistics
from concurrent.futures import ThreadPoolExecutor

import psycopg

from finrag.answer import generate
from finrag.answer.llm import LLM, Budget, Reply, cost
from finrag.eval.retrieval import Question, mcnemar_exact, pages_of, wilson
from finrag.index.embed import Embedder
from finrag.retrieve.vector import COLUMNS, Hit, search

K = 10
SETTINGS = ("unfiltered", "company_year", "gold_pages")
LABELS = ("correct", "incorrect", "refusal")
JUDGE_MODEL = "gemini-4-argon"

JUDGE_SYSTEM = """\
You grade answers to questions about companies' 10-K filings against a reference answer \
written by a financial analyst. Give one label:

- "correct": the candidate reaches the same conclusion as the reference and its key figures \
match. Figures match if they agree at the precision the question asks for or the reference \
uses; small rounding differences (about 1%) and equivalent units ($1,577 million and $1.577 \
billion) are fine. Extra detail is fine unless it contradicts the reference.
- "incorrect": a different conclusion or figure, a key part of the reference missing, or a \
claim that contradicts it.
- "refusal": the candidate declines, or says the information isn't available, without \
giving an answer.

The reference may come with a justification: use it to understand the reference, not as a \
list of things the candidate must repeat. Grade only against the reference, not against your \
own knowledge of the company."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "One or two sentences."},
        "label": {"type": "string", "enum": list(LABELS)},
    },
    "required": ["reason", "label"],
}


def grader(settings, model: str = JUDGE_MODEL, budget: Budget | None = None) -> LLM:
    return LLM(settings.gemini_api_key, model, settings.cache_dir, JUDGE_SYSTEM, JUDGE_SCHEMA,
               budget=budget)


def judge_prompt(question: str, reference: str, justification: str, candidate: str) -> str:
    return (f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Reference justification: {justification.strip() or '(none)'}\n\n"
            f"Candidate answer: {candidate}")


def gold_page_chunks(conn: psycopg.Connection, q: Question) -> list[Hit]:
    """Every chunk on the question's evidence pages, in filing order."""
    found: dict[str, Hit] = {}
    for ticker, fy, document, page in sorted(q.pages):
        for r in conn.execute(
            f"SELECT {COLUMNS}, 1.0 FROM chunks WHERE ticker = %s AND fiscal_year = %s "
            "AND document = %s AND page_start <= %s AND page_end >= %s",
            (ticker, fy, document, page, page),
        ):
            found[r[0]] = Hit(*r[:-1], score=1.0)
    return sorted(found.values(), key=lambda h: (h.document, h.id))


def contexts(questions: list[Question], embedder: Embedder, conn: psycopg.Connection,
             model: str, k: int = K) -> dict[str, dict[str, list[Hit]]]:
    """The chunks each question is answered from, per setting."""
    out = {}
    for q in questions:
        qvec = embedder.embed_query(q.question)
        out[q.id] = {
            "unfiltered": search(conn, qvec, model, k=k),
            "company_year": search(conn, qvec, model, k=k, ticker=q.ticker,
                                   fiscal_year=q.fiscal_year),
            "gold_pages": gold_page_chunks(conn, q),
        }
    return out


def _usage(r: Reply, model: str) -> dict:
    return {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "thinking_tokens": r.thinking_tokens, "ms": r.ms, "cached": r.cached,
            "usd": round(cost(model, r.input_tokens, r.output_tokens), 6)}


def answer_one(q: Question, ref: dict, hits: list[Hit], gen: LLM, judge: LLM) -> dict:
    """Answer one question from the given chunks and grade the answer."""
    a, g = generate.answer(gen, q.question, hits)
    j = judge.generate(judge_prompt(q.question, ref["answer"], ref.get("justification") or "",
                                    a["answer"]))
    grade = j.data
    gold = q.pages
    return {
        "label": grade["label"],
        "judge_reason": grade["reason"],
        **a,
        "evidence_in_context": any(pages_of(h) & gold for h in hits),
        "cited_evidence": any(pages_of(hits[c - 1]) & gold for c in a["citations"]),
        "context": [h.id for h in hits],
        "generation": _usage(g, gen.model),
        "judge": _usage(j, judge.model),
    }


def run(questions: list[Question], refs: dict[str, dict], ctx: dict, gen: LLM, judge: LLM,
        workers: int = 4) -> list[dict]:
    """Every question in every setting, `workers` API calls at a time. One row per question."""
    jobs = [(q, s) for q in questions for s in SETTINGS]
    with ThreadPoolExecutor(workers) as pool:
        done = list(pool.map(lambda js: answer_one(js[0], refs[js[0].id], ctx[js[0].id][js[1]],
                                                   gen, judge), jobs))
    rows = {q.id: {"id": q.id, "ticker": q.ticker, "fiscal_year": q.fiscal_year,
                   "question_type": q.question_type, "question": q.question} for q in questions}
    for (q, s), result in zip(jobs, done):
        rows[q.id][s] = result
    return list(rows.values())


def _p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _share(rows: list[dict], label: str) -> dict:
    n = len(rows)
    x = sum(r["label"] == label for r in rows)
    return {"n": x, "of": n, "share": round(x / n, 4) if n else None}


def report(rows: list[dict]) -> dict:
    """Labels per setting, overall, by question type and by whether the evidence was in the
    context; cost, tokens and latency; settings compared question by question."""
    out: dict = {}
    for s in SETTINGS:
        res = [r[s] for r in rows]
        n = len(res)
        correct = sum(x["label"] == "correct" for x in res)
        by_type: dict = {}
        for t in sorted({r["question_type"] for r in rows}):
            by_type[t] = _share([r[s] for r in rows if r["question_type"] == t], "correct")
        fresh = [x["generation"]["ms"] for x in res if not x["generation"]["cached"]]
        out[s] = {
            "questions": n,
            **{label: _share(res, label) for label in LABELS},
            "correct_wilson95": [round(v, 4) for v in wilson(correct, n)],
            "by_question_type": by_type,
            "evidence_in_context": _share([x for x in res if x["evidence_in_context"]],
                                          "correct"),
            "evidence_not_in_context": _share([x for x in res if not x["evidence_in_context"]],
                                              "correct"),
            "correct_citing_evidence": sum(x["cited_evidence"] for x in res
                                           if x["label"] == "correct"),
            "said_unanswerable": sum(not x["answerable"] for x in res),
            "generation": {
                "usd_total": round(sum(x["generation"]["usd"] for x in res), 4),
                "usd_per_question": round(sum(x["generation"]["usd"] for x in res) / n, 6),
                "input_tokens_mean": round(statistics.mean(
                    x["generation"]["input_tokens"] for x in res)),
                "output_tokens_mean": round(statistics.mean(
                    x["generation"]["output_tokens"] for x in res)),
                "thinking_tokens_mean": round(statistics.mean(
                    x["generation"]["thinking_tokens"] for x in res)),
                "ms_median": round(statistics.median(fresh), 1) if fresh else None,
                "ms_p95": round(_p95(fresh), 1) if fresh else None,
                "api_calls": len(fresh),
            },
            "judge_usd_total": round(sum(x["judge"]["usd"] for x in res), 4),
        }
    out["paired_correct"] = {}
    for a, b in (("unfiltered", "company_year"), ("unfiltered", "gold_pages"),
                 ("company_year", "gold_pages")):
        only_b = [r["id"] for r in rows
                  if r[b]["label"] == "correct" and r[a]["label"] != "correct"]
        only_a = [r["id"] for r in rows
                  if r[a]["label"] == "correct" and r[b]["label"] != "correct"]
        out["paired_correct"][f"{a}_vs_{b}"] = {
            f"only_{a}": only_a, f"only_{b}": only_b,
            "mcnemar_p": round(mcnemar_exact(len(only_a), len(only_b)), 4)}
    out["usd_total"] = round(sum(out[s]["generation"]["usd_total"] + out[s]["judge_usd_total"]
                                 for s in SETTINGS), 4)
    return out


def estimate(questions: list[Question], refs: dict[str, dict], ctx: dict, gen: LLM,
             judge: LLM, thinking: tuple[int, ...] = (500, 1500, 4000)) -> dict:
    """Cost of a full run before making it. Input tokens are counted exactly with the free
    count_tokens call; output is assumed: an answer of ~250 tokens and a grade of ~80, each
    plus `thinking` tokens (low, middle, high)."""
    gen_in = sum(gen.count_tokens(generate.build_prompt(q.question, ctx[q.id][s]))
                 for q in questions for s in SETTINGS)
    filler = "x " * 250  # stands in for a ~250-token candidate answer
    judge_in = sum(judge.count_tokens(judge_prompt(q.question, refs[q.id]["answer"],
                                                   refs[q.id].get("justification") or "",
                                                   filler))
                   for q in questions) * len(SETTINGS)
    calls = len(questions) * len(SETTINGS)
    return {
        "calls_each": calls,
        "generation_input_tokens": gen_in,
        "judge_input_tokens_approx": judge_in,
        "usd_by_thinking_tokens": {
            t: round(cost(gen.model, gen_in, calls * (250 + t))
                     + cost(judge.model, judge_in, calls * (80 + t)), 2)
            for t in thinking
        },
    }


def _public(row: dict) -> dict:
    """A row without FinanceBench text or text derived from it (question, answers, grader's
    reasons): labels, flags, citations, context chunk ids, tokens, cost and time."""
    out = {k: v for k, v in row.items() if k != "question"}
    for s in SETTINGS:
        out[s] = {k: v for k, v in row[s].items() if k not in ("answer", "judge_reason")}
    return out


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def format_report(summary: dict) -> str:
    header = (f"{'':<14}{'correct':>16}{'incorrect':>12}{'refusal':>10}{'95% CI':>13}"
              f"{'evidence in ctx':>18}{'$ / question':>14}{'gen ms':>9}")
    lines = [header]
    for s in SETTINGS:
        m = summary[s]
        c, ev = m["correct"], m["evidence_in_context"]
        lo, hi = m["correct_wilson95"]
        lines.append(
            f"{s:<14}{c['n']:>6}/{c['of']} {_pct(c['share']):>5}"
            f"{m['incorrect']['n']:>12}{m['refusal']['n']:>10}"
            f"{_pct(lo):>7}–{_pct(hi):<5}"
            f"{ev['n']:>10}/{ev['of']} ok"
            f"{m['generation']['usd_per_question']:>14.4f}"
            f"{m['generation']['ms_median'] or 0:>9.0f}")
    lines.append("(evidence in ctx: correct answers / questions whose context held an "
                 "evidence page)")
    lines.append("\ncorrect by question type" + "".join(f"{s:>14}" for s in SETTINGS))
    for t in summary["unfiltered"]["by_question_type"]:
        cells = "".join(f"{summary[s]['by_question_type'][t]['n']:>11}/"
                        f"{summary[s]['by_question_type'][t]['of']:<2}" for s in SETTINGS)
        lines.append(f"  {t:<22}{cells}")
    lines.append("\nquestion by question (correct in one setting only, exact McNemar p)")
    for name, p in summary["paired_correct"].items():
        a, b = [k for k in p if k.startswith("only_")]
        lines.append(f"  {name:<28} {len(p[a]):>2} vs {len(p[b]):<2} p {p['mcnemar_p']:.3f}")
    lines.append(f"\ntotal cost of this run: ${summary['usd_total']:.4f}")
    return "\n".join(lines)
