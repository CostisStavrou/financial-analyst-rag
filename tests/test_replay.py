from types import SimpleNamespace

import pytest

from finrag import cli
from finrag.agents import pipeline
from finrag.config import Company
from finrag.eval import replay
from finrag.eval.retrieval import Passage, Question
from finrag.index import embed
from tests.test_retrieval import chunk, hit

COMPANIES = [Company("Acme", "ACME", 1, "Acme", (2021, 2022))]


def plan(ticker, years):
    return pipeline.Plan(ticker=ticker, fiscal_years=years, searches=[])


def test_the_planner_filter_keeps_only_what_the_corpus_has():
    f = replay.planner_filter
    assert f(plan("ACME", [2022]), COMPANIES) == {"ticker": "ACME", "fiscal_year": 2022}
    assert f(plan("ACME", [2021, 2022]), COMPANIES) == {"ticker": "ACME", "fiscal_year": None}
    assert f(plan("ACME", [2019]), COMPANIES) == {"ticker": "ACME", "fiscal_year": None}
    # years outside the corpus go first, as in retrieve(): one year is left, so it filters
    assert f(plan("ACME", [2019, 2022]), COMPANIES) == {"ticker": "ACME", "fiscal_year": 2022}
    assert f(plan("XYZ", [2022]), COMPANIES) == {"ticker": None, "fiscal_year": None}


def test_replay_scores_sources_and_both_searches_against_the_evidence(monkeypatch):
    q = Question("q1", "ACME", 2022, "?", "metrics-generated",
                 passages=[Passage({"gold"}, {("ACME", 2022, "10-K", 5)})], not_found=0)
    gold, other = hit(chunk("gold", 5, "")), hit(chunk("other", 9, ""))
    searches = []

    def fake_search(conn, vec, model, k, ticker=None, fiscal_year=None):
        searches.append((k, ticker, fiscal_year))
        # the evidence ranks 6th unfiltered and 1st within the filing
        return [gold] + [other] * (k - 1) if ticker else [other] * 5 + [gold] + [other] * (k - 6)

    monkeypatch.setattr(replay, "search", fake_search)
    monkeypatch.setattr(replay, "retrieve", lambda *a: [other])

    class Embedder:
        def embed_query(self, text):
            return text

    rows = replay.run([q], {"q1": {"ticker": "ACME", "fiscal_years": [2022], "searches": []}},
                      {"q1": {"evidence_in_context": True}}, None, Embedder(), "m", COMPANIES)
    assert searches == [(10, None, None), (10, "ACME", 2022)]
    r = rows[0]
    assert r["evidence_in_sources_recorded"] and not r["evidence_in_sources"]
    assert r["unfiltered"] == {"hit5": False, "hit10": True}
    assert r["planner_filter"] == {"hit5": True, "hit10": True}
    s = replay.report(rows)
    assert s["agent_sources"]["change"] == {"gained": [], "lost": ["q1"], "mcnemar_p": 1.0}
    assert s["planner_filter"]["hit5"]["change"]["gained"] == ["q1"]
    assert s["planner_filter"]["filtered_by"] == {"company_and_year": 1, "company_only": 0,
                                                  "none": 0}


def test_eval_replay_builds_an_offline_embedder_before_anything_else(monkeypatch, tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "agents_runs.jsonl").write_text("")
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(data_dir=tmp_path))
    seen = {}

    def fake_make_embedder(settings, offline=False):
        seen["offline"] = offline
        raise RuntimeError("stop here")

    monkeypatch.setattr(embed, "make_embedder", fake_make_embedder)
    with pytest.raises(RuntimeError, match="stop here"):
        cli.cmd_eval_replay(None)
    assert seen == {"offline": True}
