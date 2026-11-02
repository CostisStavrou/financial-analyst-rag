import json
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from finrag.answer import generate
from finrag.answer.llm import Reply, cost
from finrag.web.app import CapReached, DailyBudget, create_app
from tests.test_retrieval import chunk, hit

HITS = [hit(chunk("ACME-1", 50, "| Trade receivables, net | 1,615.9 |")),
        hit(chunk("ACME-2", 12, "Employees: about 315,000."))]


class FakeLLM:
    """Spends like the real LLM: a saved reply is free, a fresh one checks the cap first."""

    model = "gemini-3.8-flash"

    def __init__(self, budget, saved=(), fail=False,
                 answer="Receivables were $1,615.9 million [1].", citations=(1, 7)):
        self.budget, self.saved, self.fail, self.calls = budget, set(saved), fail, 0
        self.answer, self.citations = answer, list(citations)

    def generate(self, prompt: str) -> Reply:
        question = prompt.rsplit("Question: ", 1)[1]
        data = {"answer": self.answer, "citations": self.citations, "answerable": True}
        if question in self.saved:
            return Reply(json.dumps(data), 4000, 1000, 800, 7400.0, cached=True)
        self.budget.check()
        if self.fail:
            raise ValueError("secret upstream detail")
        self.calls += 1
        self.budget.add(cost(self.model, 4000, 1000))
        return Reply(json.dumps(data), 4000, 1000, 800, 7400.0)


def client(budget=None, search=lambda q: HITS, **kw):
    budget = budget or DailyBudget(1.0)
    llm = FakeLLM(budget, **kw)
    app = create_app(search, llm, budget, companies=["Acme (2022)"],
                     examples=["What were Acme's receivables?"])
    return TestClient(app), llm


def test_the_page_and_its_info_are_served():
    c, _ = client()
    assert "Ask a 10-K" in c.get("/").text
    info = c.get("/api/info").json()
    assert info["examples"] == ["What were Acme's receivables?"]
    assert info["companies"] == ["Acme (2022)"]
    assert info["cap_usd_per_day"] == 1.0 and info["spent_today_usd"] == 0


def test_an_answer_comes_with_its_cited_pages_cost_and_time():
    c, _ = client()
    r = c.post("/api/answer", json={"question": "  What were Acme's\n receivables? "})
    assert r.status_code == 200
    d = r.json()
    assert d["question"] == "What were Acme's receivables?"
    assert d["answer"].startswith("Receivables")
    # source 7 doesn't exist, so only source 1 counts as cited
    assert [s["cited"] for s in d["sources"]] == [True, False]
    s = d["sources"][0]
    assert s["pages"] == "p. 50" and s["fiscal_year"] == 2022 and s["url"] == "u"
    assert d["usd"] == round(cost("gemini-3.8-flash", 4000, 1000), 4)
    assert d["answer_ms"] == 7400 and not d["cached"]
    assert d["spent_today_usd"] == round(cost("gemini-3.8-flash", 4000, 1000), 4)


def test_questions_must_be_3_to_300_characters_after_tidying_spaces():
    c, llm = client()
    for q in ("hi", "x" * 301, "     ", "a  "):
        assert c.post("/api/answer", json={"question": q}).status_code == 422, q
    assert llm.calls == 0


def test_a_passage_named_in_the_text_counts_as_cited():
    c, _ = client(answer="See [2] and [1, 9].", citations=())
    d = c.post("/api/answer", json={"question": "What were Acme's receivables?"}).json()
    assert [s["cited"] for s in d["sources"]] == [True, True]


def test_no_passages_means_no_gemini_call():
    c, llm = client(search=lambda q: [])
    d = c.post("/api/answer", json={"question": "What were Acme's receivables?"}).json()
    assert not d["answerable"] and d["usd"] == 0 and llm.calls == 0


def test_a_failed_search_gives_a_plain_json_error():
    def broken(q):
        raise ConnectionError("database password in here")

    c, _ = client(search=broken)
    r = c.post("/api/answer", json={"question": "What were Acme's receivables?"})
    assert r.status_code == 502 and "password" not in r.text
    assert r.json()["detail"].startswith("Something went wrong")


def test_the_daily_cap_stops_fresh_calls_but_not_saved_replies():
    budget = DailyBudget(0.005)  # one fresh call (about $0.0068) goes over it
    c, llm = client(budget, saved={"A saved question?"})
    assert c.post("/api/answer", json={"question": "A new question?"}).status_code == 200
    r = c.post("/api/answer", json={"question": "Another new question?"})
    assert r.status_code == 429 and "resets at midnight UTC" in r.json()["detail"]
    r = c.post("/api/answer", json={"question": "A saved question?"})
    assert r.status_code == 200 and r.json()["cached"]
    assert llm.calls == 1


def test_the_cap_keeps_room_for_the_worst_case_call():
    b = DailyBudget(1.0, reserve=0.3)
    b.add(0.6)
    b.check()  # 0.6 + 0.3 still fits
    b.add(0.2)
    with pytest.raises(CapReached):
        b.check()  # 0.8 + 0.3 doesn't


def test_the_real_llm_serves_saved_replies_after_the_cap(tmp_path):
    """LLM.generate looks in the reply cache before it checks the cap."""
    budget = DailyBudget(0.005)
    settings = SimpleNamespace(gemini_api_key="test-key-not-used", cache_dir=tmp_path)
    llm = generate.answerer(settings, budget=budget, max_output_tokens=4096)
    calls = []

    def fake_generate_content(model, contents, config):
        calls.append(config.max_output_tokens)
        text = json.dumps({"answer": "About 315,000 [2].", "citations": [2], "answerable": True})
        usage = SimpleNamespace(prompt_token_count=4000, candidates_token_count=200,
                                thoughts_token_count=800)
        return SimpleNamespace(text=text, usage_metadata=usage)

    llm.client = SimpleNamespace(models=SimpleNamespace(generate_content=fake_generate_content))
    c = TestClient(create_app(lambda q: HITS, llm, budget))

    def ask(q):
        return c.post("/api/answer", json={"question": q})

    assert ask("How many employees?").status_code == 200  # $0.00675: over the cap now
    assert ask("How many staff?").status_code == 429
    again = ask("How many employees?")
    assert again.status_code == 200 and again.json()["cached"]
    assert calls == [4096]


def test_the_cap_starts_again_each_day():
    days = iter([date(2026, 11, 2), date(2026, 11, 2), date(2026, 11, 3)])
    b = DailyBudget(0.01, today=lambda: next(days))
    b.add(0.02)  # still 2 Nov
    b.check()  # 3 Nov: spending starts again, so no error
    assert b.spent == 0


def test_a_failed_gemini_call_says_what_to_do_and_hides_the_detail():
    c, _ = client(fail=True)
    r = c.post("/api/answer", json={"question": "What were Acme's receivables?"})
    assert r.status_code == 502
    assert "Try again" in r.json()["detail"] and "secret" not in r.text
