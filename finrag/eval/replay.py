"""Retrieval replays of the saved agent run, with no API calls: the planner's outputs come
from data/eval/agents_runs.jsonl and every query embedding from the cache.

- agent sources: each saved plan goes through the current retrieve(), and the evidence check
  is compared with the one results/agents.json recorded for the merge that run used;
- planner filter: the question's top K (what the single call reads), searched within the
  planned company, and the planned year when the plan names exactly one, against the
  unfiltered top K.

Only retrieval is replayed: no answer is generated or graded, so these are evidence counts,
not accuracy.
"""

from __future__ import annotations

from finrag.agents.pipeline import Plan, retrieve
from finrag.config import Company
from finrag.eval.answers import K
from finrag.eval.retrieval import Question, mcnemar_exact, pages_of, wilson
from finrag.retrieve.vector import search


def _has_evidence(hits, q: Question) -> bool:
    return any(pages_of(h) & q.pages for h in hits)


def planner_filter(plan: Plan, companies: list[Company]) -> dict:
    """Company and year filters from a plan. As in retrieve(), a company or year not in the
    corpus is dropped first; the year filter is then kept only if exactly one year is left."""
    co = {c.ticker: c for c in companies}.get(plan.ticker)
    years = [y for y in plan.fiscal_years if co and y in co.fiscal_years]
    return {"ticker": co and co.ticker, "fiscal_year": years[0] if len(years) == 1 else None}


def run(questions: list[Question], plans: dict[str, dict], recorded: dict[str, dict], conn,
        embedder, model: str, companies: list[Company]) -> list[dict]:
    rows = []
    for q in questions:
        plan = Plan.model_validate(plans[q.id])
        sources = retrieve(conn, embedder, model, q.question, plan, companies)
        vec = embedder.embed_query(q.question)
        f = planner_filter(plan, companies)
        unfiltered = search(conn, vec, model, k=K)
        filtered = search(conn, vec, model, k=K, **f)
        rows.append({
            "id": q.id, "plan_years": plan.fiscal_years, "filter": f,
            "sources": len(sources),
            "evidence_in_sources_recorded": recorded[q.id]["evidence_in_context"],
            "evidence_in_sources": _has_evidence(sources, q),
            "unfiltered": {"hit5": _has_evidence(unfiltered[:5], q),
                           f"hit{K}": _has_evidence(unfiltered, q)},
            "planner_filter": {"hit5": _has_evidence(filtered[:5], q),
                               f"hit{K}": _has_evidence(filtered, q)},
        })
    return rows


def _count(rows: list[dict], get) -> dict:
    n, x = len(rows), sum(bool(get(r)) for r in rows)
    return {"n": x, "of": n, "wilson95": [round(v, 4) for v in wilson(x, n)]}


def _paired(rows: list[dict], before, after) -> dict:
    gained = [r["id"] for r in rows if after(r) and not before(r)]
    lost = [r["id"] for r in rows if before(r) and not after(r)]
    return {"gained": gained, "lost": lost,
            "mcnemar_p": round(mcnemar_exact(len(gained), len(lost)), 4)}


def report(rows: list[dict]) -> dict:
    out = {
        "agent_sources": {
            "recorded": _count(rows, lambda r: r["evidence_in_sources_recorded"]),
            "replay": _count(rows, lambda r: r["evidence_in_sources"]),
            "change": _paired(rows, lambda r: r["evidence_in_sources_recorded"],
                              lambda r: r["evidence_in_sources"]),
        },
        "planner_filter": {"filtered_by": {
            "company_and_year": sum(bool(r["filter"]["fiscal_year"]) for r in rows),
            "company_only": sum(bool(r["filter"]["ticker"]) and not r["filter"]["fiscal_year"]
                                for r in rows),
            "none": sum(not r["filter"]["ticker"] for r in rows)}},
    }
    for cut in ("hit5", f"hit{K}"):
        def before(r, cut=cut):
            return r["unfiltered"][cut]

        def after(r, cut=cut):
            return r["planner_filter"][cut]

        out["planner_filter"][cut] = {"unfiltered": _count(rows, before),
                                      "planner_filter": _count(rows, after),
                                      "change": _paired(rows, before, after)}
    return out


def format_report(s: dict) -> str:
    a, p = s["agent_sources"], s["planner_filter"]

    def ch(c: dict) -> str:
        return (f"{len(c['gained'])} gained, {len(c['lost'])} lost, exact McNemar p "
                f"{c['mcnemar_p']:.3f}")

    lines = [(f"agent sources hold the evidence: recorded run {a['recorded']['n']}/"
              f"{a['recorded']['of']}, replay {a['replay']['n']}/{a['replay']['of']} "
              f"({ch(a['change'])})"),
             (f"planner filter: company and year {p['filtered_by']['company_and_year']}, "
              f"company only {p['filtered_by']['company_only']}, none "
              f"{p['filtered_by']['none']}")]
    for cut in ("hit5", f"hit{K}"):
        m = p[cut]
        lines.append(f"single-call {cut}: unfiltered {m['unfiltered']['n']}, planner filter "
                     f"{m['planner_filter']['n']} of {m['unfiltered']['of']} ({ch(m['change'])})")
    return "\n".join(lines)
