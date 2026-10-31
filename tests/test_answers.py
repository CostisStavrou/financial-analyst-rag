import json

from finrag.answer import generate
from finrag.answer.llm import LLM, Reply, ReplyCache, cost
from finrag.eval import answers
from finrag.eval.retrieval import Passage, Question
from tests.test_retrieval import chunk, hit


def _hits():
    return [hit(chunk("a", 5, "| Net sales | 34,229 |")), hit(chunk("b", 9, "Text."))]


def test_build_prompt_numbers_sources_and_ends_with_the_question():
    p = generate.build_prompt("What were net sales?", _hits())
    assert p.startswith("Sources:\n\n[1] Acme 10-K FY2022, Item 8., p. 5\n| Net sales | 34,229 |")
    assert "\n\n[2] Acme 10-K FY2022, Item 8., p. 9\nText." in p
    assert p.endswith("Question: What were net sales?")


def test_parse_keeps_only_real_citations():
    reply = Reply(json.dumps({"answer": "34,229", "citations": [2, 2, 0, 7, "1"],
                              "answerable": True}), 1, 1, 0, 1.0)
    assert generate.parse(reply, 2) == {"answer": "34,229", "citations": [2],
                                        "answerable": True}
    assert generate.parse(Reply("{}", 1, 1, 0, 1.0), 2) == {
        "answer": "", "citations": [], "answerable": True}


def test_cost():
    assert cost("gemini-3.8-flash", 1_000_000, 0) == 0.75
    assert cost("gemini-3.8-flash", 0, 1_000_000) == 3.75
    assert round(cost("gemini-4-argon", 1000, 500), 6) == round((1000 * 2 + 500 * 10) / 1e6, 6)


def test_cached_reply_costs_no_call(tmp_path):
    llm = LLM("not-a-real-key", "gemini-3.8-flash", tmp_path, "system", {"type": "object"})
    reply = Reply('{"answer": "x"}', 100, 40, 30, 1234.0)
    llm.cache.put(llm._key("prompt"), reply)
    got = llm.generate("prompt")
    assert got.cached and (got.input_tokens, got.output_tokens, got.ms) == (100, 40, 1234.0)
    assert llm.calls == 0
    # any change to the settings is another cache entry
    other = LLM("not-a-real-key", "gemini-3.8-flash", tmp_path, "system 2", {"type": "object"})
    assert other._key("prompt") != llm._key("prompt")
    assert ReplyCache(tmp_path / "replies.sqlite").get(llm._key("prompt")).text == reply.text


def test_judge_prompt():
    p = answers.judge_prompt("Q?", "$1,577", "", "It was $1.577 billion.")
    assert "Reference answer: $1,577" in p and "Reference justification: (none)" in p
    assert p.endswith("Candidate answer: It was $1.577 billion.")


class FakeLLM:
    def __init__(self, model, replies):
        self.model, self.replies, self.prompts = model, replies, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _question(qid="q1", qtype="metrics-generated"):
    return Question(qid, "ACME", 2022, "What were net sales?", qtype,
                    passages=[Passage({"a"}, {("ACME", 2022, "10-K", 5)})], not_found=0)


def test_answer_one_grades_and_flags_evidence():
    gen = FakeLLM("gemini-3.8-flash", [Reply(json.dumps(
        {"answer": "$34,229 million [1]", "citations": [1], "answerable": True}),
        3000, 900, 700, 5000.0)])
    judge = FakeLLM("gemini-4-argon", [Reply(json.dumps(
        {"reason": "Same figure.", "label": "correct"}), 500, 300, 250, 3000.0)])
    r = answers.answer_one(_question(), {"answer": "$34229", "justification": None}, _hits(),
                           gen, judge)
    assert r["label"] == "correct" and r["evidence_in_context"] and r["cited_evidence"]
    assert r["context"] == ["a", "b"]
    assert r["generation"]["usd"] == round(cost("gemini-3.8-flash", 3000, 900), 6)
    assert "Candidate answer: $34,229 million [1]" in judge.prompts[0]
    # citing only the other page: evidence was in the context but not cited
    gen.replies.append(Reply(json.dumps({"answer": "?", "citations": [2], "answerable": True}),
                             1, 1, 0, 1.0))
    judge.replies.append(Reply(json.dumps({"reason": "", "label": "incorrect"}), 1, 1, 0, 1.0))
    r = answers.answer_one(_question(), {"answer": "$34229"}, _hits(), gen, judge)
    assert r["evidence_in_context"] and not r["cited_evidence"]


def _row(qid, qtype, labels, cached=False):
    def res(label):
        usage = {"input_tokens": 3000, "output_tokens": 800, "thinking_tokens": 600,
                 "ms": 4000.0, "cached": cached, "usd": 0.005}
        return {"label": label, "judge_reason": "r", "answer": "a", "citations": [1],
                "answerable": label != "refusal", "evidence_in_context": label != "refusal",
                "cited_evidence": label == "correct", "context": ["a"],
                "generation": usage, "judge": {**usage, "usd": 0.004}}

    return {"id": qid, "ticker": "ACME", "fiscal_year": 2022, "question_type": qtype,
            "question": "Q?", **{s: res(lab) for s, lab in zip(answers.SETTINGS, labels)}}


def test_report_and_public_rows():
    rows = [_row("q1", "metrics-generated", ("correct", "correct", "correct")),
            _row("q2", "domain-relevant", ("incorrect", "correct", "correct")),
            _row("q3", "domain-relevant", ("refusal", "incorrect", "correct"))]
    s = answers.report(rows)
    u = s["unfiltered"]
    assert (u["correct"]["n"], u["incorrect"]["n"], u["refusal"]["n"]) == (1, 1, 1)
    assert u["by_question_type"]["domain-relevant"] == {"n": 0, "of": 2, "share": 0.0}
    assert u["evidence_not_in_context"] == {"n": 0, "of": 1, "share": 0.0}
    assert u["said_unanswerable"] == 1 and u["correct_citing_evidence"] == 1
    assert u["generation"]["usd_total"] == 0.015 and u["generation"]["api_calls"] == 3
    assert s["gold_pages"]["correct"]["share"] == 1.0
    p = s["paired_correct"]["unfiltered_vs_gold_pages"]
    assert (p["only_unfiltered"], p["only_gold_pages"]) == ([], ["q2", "q3"])
    assert s["usd_total"] == round(3 * (0.015 + 0.012), 4)
    assert "total cost of this run" in answers.format_report(s)
    out = answers._public(rows[0])
    assert "question" not in out
    assert "answer" not in out["unfiltered"] and "judge_reason" not in out["unfiltered"]
    assert out["unfiltered"]["label"] == "correct"


class PageConn:
    def __init__(self, by_page):
        self.by_page = by_page

    def execute(self, sql, params):
        return self.by_page.get(params[3], [])


def test_gold_page_chunks_in_filing_order_without_repeats():
    def row(c):
        h = hit(c)
        return (h.id, h.ticker, h.company, h.fiscal_year, h.document, h.section, h.page_start,
                h.page_end, h.page_label, h.chunk_type, h.content, h.source_url, 1.0)

    spans = chunk("ACME-0009", 5, "", page_end=6)  # on both evidence pages
    conn = PageConn({5: [row(chunk("ACME-0010", 5, "")), row(spans)], 6: [row(spans)]})
    q = Question("q", "ACME", 2022, "?", "x",
                 passages=[Passage(set(), {("ACME", 2022, "10-K", 5), ("ACME", 2022, "10-K", 6)})],
                 not_found=0)
    assert [h.id for h in answers.gold_page_chunks(conn, q)] == ["ACME-0009", "ACME-0010"]


def test_budget_stops_calls_once_spent(tmp_path):
    from finrag.answer.llm import Budget

    budget = Budget(0.01)
    llm = LLM("not-a-real-key", "gemini-3.8-flash", tmp_path, "s", {"type": "object"},
              budget=budget)
    llm.cache.put(llm._key("cached"), Reply("{}", 1, 1, 0, 1.0))
    budget.add(0.01)
    assert llm.generate("cached").cached  # the cache still answers
    try:
        llm.generate("new prompt")
    except RuntimeError as e:
        assert "cap of $0.01 reached" in str(e) and llm.calls == 0
    else:
        raise AssertionError("expected the cap to stop the call")
