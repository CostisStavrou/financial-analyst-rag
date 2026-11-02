import json

from finrag.agents import pipeline
from finrag.agents.pipeline import Plan, calculate, retrieve
from finrag.config import Company
from finrag.eval import agents
from tests.test_retrieval import chunk, hit


def test_calculator_does_arithmetic_and_nothing_else():
    assert calculate("(34229 - 31750) / 31750 * 100")["result"] == round(
        (34229 - 31750) / 31750 * 100, 10)
    assert calculate("1,577 + 3")["result"] == 1580  # thousands separators are dropped
    assert calculate("(1,577 - 1,000) / 2")["result"] == 288.5
    assert calculate("round(2 ** 0.5, 3) + abs(-1) + max(1, 2)")["result"] == 4.414
    assert calculate("max(100, 200)")["result"] == 200
    # inside a call, 100,200 could be two arguments or one number: refused, never guessed
    for ambiguous in ("max(100,200)", "max(100,200, 5)", "round(1,577 / 3, 2)"):
        assert "ambiguous" in calculate(ambiguous)["error"], ambiguous
    # one-argument functions can't be ambiguous; nor can grouping parentheses
    assert calculate("abs(-1,577) + sqrt(1,000,000) + (2,000)")["result"] == 4577
    # a comma after a decimal, an exponent or a 4+ digit run is not a thousands separator
    for bad in ("0.5,123", "2.5,000 * 2", "1e3,000", "12345,678", "1,5000"):
        assert "error" in calculate(bad), bad
    for bad in ("__import__('os').system('ls')", "x + 1", "1 / 0", "9 ** 9 ** 9",
                "open('f')", "(1).real"):
        assert "error" in calculate(bad), bad


COMPANIES = [Company("Acme", "ACME", 1, "Acme", (2021, 2022))]


def test_retrieve_filters_to_the_plan_and_merges(monkeypatch):
    calls = []

    def fake_search(conn, vec, model, k, ticker=None, fiscal_year=None):
        calls.append((vec, k, ticker, fiscal_year))
        return [hit(chunk(f"{vec}-{fiscal_year}-{i}", i, "")) for i in range(k)] + [
            hit(chunk("shared", 99, ""))]

    class Emb:
        def embed_query(self, text):
            return text

    monkeypatch.setattr(pipeline, "search", fake_search)
    plan = Plan(ticker="ACME", fiscal_years=[2022, 2019], searches=["a", "b"])
    hits = retrieve(None, Emb(), "m", "Q", plan, COMPANIES)
    # 2019 isn't in the corpus, so only 2022; the question at top 10, each search at top 5
    assert calls == [("Q", 10, "ACME", 2022), ("a", 5, "ACME", 2022), ("b", 5, "ACME", 2022)]
    ids = [h.id for h in hits]
    assert len(ids) == len(set(ids)) == pipeline.MAX_SOURCES
    assert ids[:6] == ["Q-2022-0", "a-2022-0", "b-2022-0", "Q-2022-1", "a-2022-1", "b-2022-1"]
    calls.clear()
    retrieve(None, Emb(), "m", "Q", Plan(ticker="ZZZ", fiscal_years=[2022], searches=[]),
             COMPANIES)
    assert calls == [("Q", 10, None, None)]  # unknown company: no filter at all


def test_retrieve_keeps_every_year_and_search_under_the_cap(monkeypatch):
    # The old merge took the question's lists first: with 2 years they alone filled the cap
    # of 20 and every planned search was dropped; with 3 the last year was cut.
    def fake_search(conn, vec, model, k, ticker=None, fiscal_year=None):
        return [hit(chunk(f"{vec}-{fiscal_year}-{i}", i, "", fy=fiscal_year)) for i in range(k)]

    class Emb:
        def embed_query(self, text):
            return text

    monkeypatch.setattr(pipeline, "search", fake_search)
    companies = [Company("Acme", "ACME", 1, "Acme", (2020, 2021, 2022))]
    for years in ([2021, 2022], [2020, 2021, 2022]):
        plan = Plan(ticker="ACME", fiscal_years=years, searches=["a", "b", "c"])
        hits = retrieve(None, Emb(), "m", "Q", plan, companies)
        assert len(hits) == pipeline.MAX_SOURCES
        assert {h.fiscal_year for h in hits} == set(years)
        assert {h.id.split("-")[0] for h in hits} == {"Q", "a", "b", "c"}
        for text in "Qabc":  # every (query, year) list gives at least its top hit
            for fy in years:
                assert f"{text}-{fy}-0" in {h.id for h in hits}


def test_a_run_never_resumes_from_runs_made_with_another_merge(tmp_path):
    runs = tmp_path / "runs.jsonl"
    runs.write_text('{"id": "q1", "plan": null}\n')  # first run: no merge recorded
    try:
        agents.run([], None, runs)
    except RuntimeError as e:
        assert "--tag" in str(e)
    else:
        raise AssertionError("resumed from runs made with another merge")
    runs.write_text(json.dumps({"id": "q1", "merge": pipeline.MERGE}) + "\n")
    assert list(agents.run([], None, runs)) == ["q1"]


def _row(qid, label, draft, changed=None, ticker_right=True):
    step = {"input_tokens": 1000, "output_tokens": 200, "thinking_tokens": 100,
            "model_calls": 2, "tool_calls": 1, "usd": 0.002, "ms": 3000.0}
    return {"id": qid, "question_type": "metrics-generated", "question": "Q?",
            "plan_ticker_right": ticker_right, "plan_year_right": True, "sources": ["a", "b"],
            "evidence_in_context": True, "cited_evidence": label == "correct",
            "draft_label": draft, "label": label, "draft_reason": "r", "judge_reason": "r",
            "draft_answer": "d", "answer": "a",
            "verifier_changed": changed if changed is not None else label != draft,
            "verifier_problems": [], "answerable": True,
            "steps": {"planner": step, "analyst": step, "verifier": step,
                      "retriever": {"ms": 300.0, "sources": 2}},
            "ms": 9300.0, "usd": 0.006}


def test_report_measures_the_verifier_and_compares_with_the_single_call():
    rows = [_row("q1", "correct", "incorrect"), _row("q2", "incorrect", "correct"),
            _row("q3", "correct", "correct", ticker_right=False)]
    s = agents.report(rows, baseline={"q1": "incorrect", "q2": "correct", "q3": "correct"})
    assert s["correct"]["n"] == 2 and s["draft_correct"]["n"] == 2
    assert s["verifier"] == {"changed": 2, "fixed": ["q1"], "broke": ["q2"]}
    assert s["plan_ticker_right"] == 2
    b = s["vs_single_call_no_filter"]
    assert (b["only_agents"], b["only_single_call"]) == (["q1"], ["q2"])
    assert s["steps"]["analyst"]["usd_per_question"] == 0.002
    assert s["steps"]["retriever"]["sources_mean"] == 2
    assert "verifier changed 2, fixed 1, broke 1" in agents.format_report(s)
    out = agents.public(rows[0])
    assert not {"question", "answer", "draft_answer", "judge_reason"} & out.keys()
