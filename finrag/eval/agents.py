"""The multi-agent pipeline on FinanceBench, scored like the single-call answers.

Each in-scope question goes through finrag.agents.pipeline once, with no filter given: the
planner has to find the company and year itself. The final answer, and the analyst's draft
before the verifier, are graded by the same grader and rubric as finrag.eval.answers, so the
results compare with its no-filter row question by question.

Runs are saved one question at a time to data/eval/agents_runs.jsonl, so a stopped run goes on
where it left off and finished questions are never paid for twice.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path

from finrag.agents.pipeline import MERGE
from finrag.answer.llm import LLM
from finrag.eval.answers import _p95, _share, judge_prompt
from finrag.eval.retrieval import Question, mcnemar_exact, wilson

STEPS = ("planner", "retriever", "analyst", "verifier")


def _hit_row(h) -> list:
    return [h.id, h.ticker, h.fiscal_year, h.document, h.page_start, h.page_end]


def run(questions: list[Question], pipeline, runs_path: Path, max_usd: float | None = None
        ) -> dict[str, dict]:
    """Run every question not yet in runs_path; returns all runs by question id."""
    done = {}
    if runs_path.exists():
        done = {r["id"]: r for r in map(json.loads, runs_path.read_text().splitlines())}
        if stale := sorted(i for i, r in done.items() if r.get("merge") != MERGE):
            raise RuntimeError(f"{runs_path} holds {len(stale)} runs made with another retrieval "
                               "merge; start a new run with --tag instead of mixing them.")
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    async def all_questions():  # one event loop for the whole run
        spent = 0.0
        with runs_path.open("a") as f:
            for q in questions:
                if q.id in done:
                    continue
                if max_usd is not None and spent >= max_usd:
                    raise RuntimeError(f"Spending cap of ${max_usd:.2f} reached (${spent:.4f} "
                                       "spent); finished questions are saved.")
                r = await pipeline.run(q.question)
                row = {"id": q.id, "merge": MERGE,
                       **{k: v for k, v in r.items() if k != "sources"},
                       "sources": [_hit_row(h) for h in r["sources"]]}
                f.write(json.dumps(row) + "\n")
                f.flush()
                spent += r["usd"]
                done[q.id] = row

    asyncio.run(all_questions())
    return done


def _pages(src: list) -> set:
    _id, ticker, fy, doc, p0, p1 = src
    return {(ticker, fy, doc, p) for p in range(p0, p1 + 1)}


def grade(questions: list[Question], refs: dict, runs: dict, judge: LLM) -> list[dict]:
    """One result row per question: plan checks, retrieval, draft and final answers graded."""
    rows = []
    for q in questions:
        r = runs[q.id]
        plan, draft, final = r["plan"] or {}, r["draft"] or {}, r["verdict"] or {}
        ref = refs[q.id]
        labels = {}
        for name, a in (("draft", draft), ("final", final)):
            j = judge.generate(judge_prompt(q.question, ref["answer"],
                                            ref.get("justification") or "", a.get("answer", "")))
            labels[name] = (j.data["label"], j.data["reason"])
        n = len(r["sources"])
        cited = [c for c in final.get("citations", []) if isinstance(c, int) and 1 <= c <= n]
        rows.append({
            "id": q.id, "ticker": q.ticker, "fiscal_year": q.fiscal_year,
            "question_type": q.question_type, "question": q.question,
            "plan_ticker_right": plan.get("ticker") == q.ticker,
            "plan_year_right": q.fiscal_year in (plan.get("fiscal_years") or []),
            "plan_years": plan.get("fiscal_years"),
            "searches": len(plan.get("searches") or []),
            "sources": [s[0] for s in r["sources"]],
            "evidence_in_context": any(_pages(s) & q.pages for s in r["sources"]),
            "cited_evidence": any(_pages(r["sources"][c - 1]) & q.pages for c in cited),
            "draft_label": labels["draft"][0], "draft_reason": labels["draft"][1],
            "label": labels["final"][0], "judge_reason": labels["final"][1],
            "draft_answer": draft.get("answer", ""), "answer": final.get("answer", ""),
            "draft_missing": not r["draft"],
            "verifier_changed": final.get("answer", "") != draft.get("answer", ""),
            "verifier_problems": final.get("problems", []),
            "answerable": final.get("answerable", True),
            "steps": r["steps"], "ms": r["ms"], "usd": r["usd"],
        })
    return rows


def report(rows: list[dict], baseline: dict[str, str] | None = None) -> dict:
    """Accuracy, plan and retrieval checks, the verifier's effect, cost and time per step, and
    the comparison with the single-call answers (baseline: question id -> label)."""
    n = len(rows)
    correct = sum(r["label"] == "correct" for r in rows)
    changed = [r for r in rows if r["verifier_changed"]]
    out: dict = {
        "questions": n,
        **{label: _share(rows, label) for label in ("correct", "incorrect", "refusal")},
        "correct_wilson95": [round(v, 4) for v in wilson(correct, n)],
        "draft_correct": _share([{"label": r["draft_label"]} for r in rows], "correct"),
        "verifier": {
            "changed": len(changed),
            "fixed": [r["id"] for r in changed
                      if r["label"] == "correct" and r["draft_label"] != "correct"],
            "broke": [r["id"] for r in changed
                      if r["draft_label"] == "correct" and r["label"] != "correct"],
        },
        "draft_missing": [r["id"] for r in rows if r.get("draft_missing")],
        "plan_ticker_right": sum(r["plan_ticker_right"] for r in rows),
        "plan_year_right": sum(r["plan_year_right"] for r in rows),
        "evidence_in_context": _share([r for r in rows if r["evidence_in_context"]], "correct"),
        "evidence_not_in_context": _share([r for r in rows if not r["evidence_in_context"]],
                                          "correct"),
        "sources_mean": round(statistics.mean(len(r["sources"]) for r in rows), 1),
        "correct_citing_evidence": sum(r["cited_evidence"] for r in rows
                                       if r["label"] == "correct"),
        "by_question_type": {t: _share([r for r in rows if r["question_type"] == t], "correct")
                             for t in sorted({r["question_type"] for r in rows})},
        "steps": {},
        "usd_per_question": round(statistics.mean(r["usd"] for r in rows), 6),
        "usd_total": round(sum(r["usd"] for r in rows), 4),
        "ms_median": round(statistics.median(r["ms"] for r in rows), 1),
        "ms_p95": round(_p95([r["ms"] for r in rows]), 1),
    }
    for step in STEPS:
        s = [r["steps"][step] for r in rows if step in r["steps"]]
        ms = [x["ms"] for x in s if x.get("ms") is not None]
        out["steps"][step] = {
            "ms_median": round(statistics.median(ms), 1) if ms else None,
            "ms_p95": round(_p95(ms), 1) if ms else None,
            **({"usd_per_question": round(statistics.mean(x["usd"] for x in s), 6),
                "input_tokens_mean": round(statistics.mean(x["input_tokens"] for x in s)),
                "output_tokens_mean": round(statistics.mean(x["output_tokens"] for x in s)),
                "model_calls_mean": round(statistics.mean(x["model_calls"] for x in s), 2),
                "tool_calls_mean": round(statistics.mean(x["tool_calls"] for x in s), 2)}
               if step != "retriever" else
               {"sources_mean": round(statistics.mean(x["sources"] for x in s), 1)}),
        }
    if baseline:
        only_agents = [r["id"] for r in rows
                       if r["label"] == "correct" and baseline.get(r["id"]) != "correct"]
        only_single = [r["id"] for r in rows
                       if baseline.get(r["id"]) == "correct" and r["label"] != "correct"]
        out["vs_single_call_no_filter"] = {
            "only_agents": only_agents, "only_single_call": only_single,
            "mcnemar_p": round(mcnemar_exact(len(only_agents), len(only_single)), 4)}
    return out


def public(row: dict) -> dict:
    """A row without FinanceBench text or text derived from it."""
    drop = ("question", "draft_reason", "judge_reason", "draft_answer", "answer",
            "verifier_problems")
    return {k: v for k, v in row.items() if k not in drop}


def _pct(x):
    return "–" if x is None else f"{x:.0%}"


def format_report(s: dict) -> str:
    c, lo_hi = s["correct"], s["correct_wilson95"]
    v = s["verifier"]
    lines = [
        (f"correct {c['n']}/{c['of']} ({_pct(c['share'])}, 95% CI {_pct(lo_hi[0])}–"
         f"{_pct(lo_hi[1])}); incorrect {s['incorrect']['n']}, refusals {s['refusal']['n']}"),
        (f"draft before the verifier: {s['draft_correct']['n']}/{c['of']} correct; verifier "
         f"changed {v['changed']}, fixed {len(v['fixed'])}, broke {len(v['broke'])}"),
        (f"planner: company right {s['plan_ticker_right']}/{c['of']}, year right "
         f"{s['plan_year_right']}/{c['of']}; sources per question {s['sources_mean']}"),
        (f"evidence page in the sources: {s['evidence_in_context']['of']}/{c['of']} "
         f"({s['evidence_in_context']['n']} of them correct)"),
    ]
    if "vs_single_call_no_filter" in s:
        b = s["vs_single_call_no_filter"]
        lines.append(f"vs single call, no filter: only agents {len(b['only_agents'])}, only "
                     f"single call {len(b['only_single_call'])}, exact McNemar p "
                     f"{b['mcnemar_p']:.3f}")
    lines.append(f"\n{'step':<11}{'ms med':>9}{'p95':>9}{'$ / q':>10}{'in tok':>8}{'out tok':>9}"
                 f"{'calls':>7}{'tools':>7}")
    for step, m in s["steps"].items():
        if step == "retriever":
            lines.append(f"{step:<11}{m['ms_median'] or 0:>9.0f}{m['ms_p95'] or 0:>9.0f}"
                         f"{'':>10}{'':>8}{'':>9}  sources {m['sources_mean']}")
            continue
        lines.append(f"{step:<11}{m['ms_median'] or 0:>9.0f}{m['ms_p95'] or 0:>9.0f}"
                     f"{m['usd_per_question']:>10.4f}{m['input_tokens_mean']:>8}"
                     f"{m['output_tokens_mean']:>9}{m['model_calls_mean']:>7}"
                     f"{m['tool_calls_mean']:>7}")
    lines.append(f"\nper question: ${s['usd_per_question']:.4f}, {s['ms_median'] / 1000:.1f} s "
                 f"median ({s['ms_p95'] / 1000:.1f} s p95); agents total ${s['usd_total']:.4f}")
    return "\n".join(lines)
