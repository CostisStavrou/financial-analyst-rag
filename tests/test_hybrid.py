import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction

import psycopg
import pytest

from finrag.eval.retrieval import (
    SETTINGS,
    _public,
    check_keyword_index,
    format_report,
    mcnemar_exact,
    report,
    same_as_baseline,
    wilson,
)
from finrag.index.db import KEYWORD_VERSION
from finrag.retrieve import keyword
from finrag.retrieve.hybrid import competition_ranks, fuse, md5, rrf
from tests.test_retrieval import chunk, hit


def test_competition_ranks():
    assert competition_ranks([]) == []
    assert competition_ranks([9.0, 7.5, 7.5, 3.0, 3.0, 1.0]) == [1, 2, 2, 4, 4, 6]


def test_rrf_maths_and_union():
    fused = rrf({"vector": [("a", 1), ("b", 2), ("c", 3)], "keyword": [("c", 1), ("d", 2)]})
    by_id = {f.id: f for f in fused}
    assert set(by_id) == {"a", "b", "c", "d"}
    assert by_id["c"].score == float(Fraction(1, 63) + Fraction(1, 61))
    assert by_id["c"].ranks == {"vector": 3, "keyword": 1}
    assert by_id["d"].ranks == {"vector": None, "keyword": 2}
    assert fused[0].id == "c"  # found by both legs


def test_rrf_agreement_beats_any_single_leg():
    # at depth 50 and k 60, a chunk in both lists beats a chunk ranked 1st in only one
    legs = {"vector": [("top", 1)] + [(f"v{i}", i) for i in range(2, 50)] + [("both", 50)],
            "keyword": [(f"k{i}", i) for i in range(1, 50)] + [("both", 50)]}
    assert rrf(legs)[0].id == "both"


def test_rrf_ties_go_to_the_vector_leg_whatever_the_id():
    for v, k in (("zzz", "aaa"), ("aaa", "zzz")):
        fused = rrf({"vector": [(v, 3)], "keyword": [(k, 3)]})
        assert fused[0].score == fused[1].score
        assert [f.id for f in fused] == [v, k]
    # same ranks in both legs, mirrored: the better vector rank wins
    fused = rrf({"vector": [("x", 1), ("y", 2)], "keyword": [("y", 1), ("x", 2)]})
    assert [f.id for f in fused] == ["x", "y"]
    # flipping the leg order flips the tie-break (the sensitivity check does this)
    fused = rrf({"keyword": [("y", 1), ("x", 2)], "vector": [("x", 1), ("y", 2)]})
    assert [f.id for f in fused] == ["y", "x"]
    # a full tie is settled by md5(id), not by id
    a, b = sorted(["MMM-FY2018-10K-0001", "MMM-FY2022-10K-0001"], key=md5)
    fused = rrf({"vector": [], "keyword": [(b, 1), (a, 1)]})
    assert [f.id for f in fused] == [a, b]


def test_fuse_shares_ranks_between_equal_scores():
    same_text_two_years = [hit(chunk("old", 5, "")), hit(chunk("new", 5, ""))]
    fused = fuse({"vector": [], "keyword": same_text_two_years})  # both score 0.5
    assert [f.ranks["keyword"] for f in fused] == [1, 1]
    vec = [hit(chunk("v", 1, ""))]
    assert [f.id for f in fuse({"vector": vec, "keyword": []})] == ["v"]  # no keyword hits


def test_wilson_and_mcnemar():
    lo, hi = wilson(49, 57)
    assert (round(lo, 3), round(hi, 3)) == (0.747, 0.927)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(6, 0) == mcnemar_exact(0, 6) == 2 / 64
    assert mcnemar_exact(5, 0) == 2 / 32
    assert round(mcnemar_exact(8, 1), 4) == 0.0391
    assert mcnemar_exact(3, 3) == 1.0


def test_keyword_query_filters():
    sql, params = keyword._query("capex?", 50, None, None, None)
    assert "WHERE" not in sql.split("scored AS")[1].split("GROUP BY")[0]
    assert params["k"] == 50 and params["question"] == "capex?"
    sql, params = keyword._query("capex?", 50, "MMM", 2018, "8")
    where = sql.split("scored AS")[1].split("GROUP BY")[0]
    assert "d.ticker = %(ticker)s AND d.fiscal_year = %(fy)s AND d.item = %(item)s" in where
    assert (params["ticker"], params["fy"], params["item"]) == ("MMM", 2018, "8")
    assert "md5(p.id)" in sql  # ties: not by id


class Rows(list):
    def fetchone(self):
        return self[0] if self else None


class KeywordConn:
    """keyword_docs rows, and index_meta holding the keyword index version."""

    def __init__(self, chunks, version=KEYWORD_VERSION, error=None):
        self.rows = Rows((c.id, hashlib.md5(c.embed_text().encode()).hexdigest())
                         for c in chunks)
        self.version, self.error = version, error

    def execute(self, sql, params=()):
        if "keyword_docs" in sql and self.error:
            raise self.error
        if "to_regclass" in sql:
            return Rows([("index_meta",)])
        if "index_meta" in sql:
            return Rows([(self.version,)])
        return self.rows


def test_check_keyword_index():
    files = [chunk("c0", 1, "alpha"), chunk("c1", 2, "beta")]
    check_keyword_index(KeywordConn(files), files)
    renamed = [chunk("c0", 1, "alpha"), chunk("c1", 3, "beta")]  # header now says page 3
    with pytest.raises(RuntimeError, match="1 chunks with other text in the keyword index"):
        check_keyword_index(KeywordConn(files), renamed)
    with pytest.raises(RuntimeError, match="1 chunks missing from the keyword index"):
        check_keyword_index(KeywordConn(files[:1]), files)
    with pytest.raises(RuntimeError, match="version 0, this code expects"):
        check_keyword_index(KeywordConn(files, version="0"), files)
    # a database from before the keyword views
    with pytest.raises(RuntimeError, match="missing or empty. Run `finrag index`"):
        check_keyword_index(KeywordConn(files, error=psycopg.errors.UndefinedTable()), files)


def _setting(page_rank, legs_ranks, hnsw=True):
    def s(rank):
        return {"page_rank": rank, "chunk_rank": rank, "passage_ranks": [rank],
                "all_rank": rank, "search_ms": 5.0}

    return {**s(page_rank), "used_hnsw": hnsw, "overlap": 4,
            "retrieved": [["c1", 5, 5, 0.032787, 1, 2]],
            "legs": {leg: {**s(r), "page_rank_depth": r, "retrieved": ["c1", "c2"]}
                     for leg, r in zip(("vector", "keyword"), legs_ranks)}}


def _hybrid_rows():
    # q1: hybrid fixes hit@1; q2: hybrid breaks hit@5; q3: all three miss
    cases = {"q1": (1, (2, 1)), "q2": (8, (4, None)), "q3": (None, (None, None))}
    return [{"id": qid, "question": "?", "question_type": "novel-generated", "ticker": "ACME",
             "embed_ms": 300.0, "embed_cached": True,
             **{setting: _setting(rank, legs) for setting in SETTINGS}}
            for qid, (rank, legs) in cases.items()]


def test_report_for_hybrid_runs():
    rows = _hybrid_rows()
    summary = report(rows, k=20)
    u = summary["unfiltered"]
    assert u["page_hit@1"] == round(1 / 3, 4) and u["legs"]["vector"]["page_hit@1"] == 0
    assert u["legs"]["keyword"]["page_hit@5"] == round(1 / 3, 4)
    assert u["vs_vector"]["page_hit@1"]["fixed"] == ["q1"]
    assert u["vs_vector"]["page_hit@5"]["broken"] == ["q2"]
    assert u["vs_vector"]["page_hit@20"] == {"fixed": [], "broken": [], "mcnemar_p": 1.0}
    assert u["page_hit@5_wilson95"]["vector"] == [round(x, 4) for x in wilson(2, 3)]
    assert u["legs_overlap_median"] == 4
    text = format_report(summary, k=20)
    assert "  keyword" in text and "1/0 p1.00" in text and "3/3" in text


def test_public_rows_keep_leg_ranks_without_lists():
    out = _public(_hybrid_rows()[0])
    assert "question" not in out
    assert out["unfiltered"]["retrieved"] == [["c1", 1, 2]]
    assert "retrieved" not in out["unfiltered"]["legs"]["vector"]
    assert out["unfiltered"]["legs"]["keyword"]["page_rank_depth"] == 1


def test_same_as_baseline(tmp_path):
    rows = _hybrid_rows()
    base = {"questions": [{"id": r["id"], **{s: {"retrieved": ["c1"]} for s in SETTINGS}}
                          for r in rows]}
    base["questions"][1]["company"]["retrieved"] = ["c2"]
    path = tmp_path / "retrieval_vector.json"
    path.write_text(json.dumps(base))
    same = same_as_baseline(rows, path)
    assert (same["identical"], same["searches"], same["differ"]) == (8, 9, ["q2 company"])
    assert same_as_baseline(rows, tmp_path / "missing.json") is None


# --- BM25 in SQL against a plain-Python BM25, on the real keyword index (skipped without it)

def _db():
    from finrag.config import get_settings
    from finrag.index import db

    try:
        conn = db.connect(get_settings().database_url)
        conn.execute("SELECT 1 FROM keyword_docs LIMIT 1")
        return conn
    except (psycopg.OperationalError, psycopg.errors.ObjectNotInPrerequisiteState,
            psycopg.errors.UndefinedTable):
        return None


@pytest.fixture(scope="module")
def conn():
    c = _db()
    if c is None:
        pytest.skip("no populated keyword index (run docker compose up -d and finrag index)")
    yield c
    c.close()


def bm25_reference(conn, question, ticker=None, fy=None):
    """Lucene BM25 computed in Python from the postings, one term at a time."""
    docs = {r[0]: r[1:] for r in conn.execute("SELECT id, ticker, fiscal_year, dl "
                                              "FROM keyword_docs")}
    n = len(docs)
    avgdl = sum(d[2] for d in docs.values()) / n
    scores: dict[str, float] = defaultdict(float)
    terms = conn.execute("SELECT lexeme, cardinality(positions) "
                         "FROM unnest(to_tsvector('english', %s))", (question,)).fetchall()
    for lexeme, qtf in terms:
        postings = conn.execute("SELECT id, tf FROM keyword_postings WHERE lexeme = %s",
                                (lexeme,)).fetchall()
        df = len(postings)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for cid, tf in postings:
            t, y, dl = docs[cid]
            if (ticker and t != ticker) or (fy and y != fy):
                continue
            scores[cid] += qtf * idf * tf * (keyword.K1 + 1) / (
                tf + keyword.K1 * (1 - keyword.B + keyword.B * dl / avgdl))
    return sorted(scores.items(), key=lambda x: -x[1])


# our own questions (FinanceBench's text is CC BY-NC and is kept out of the repo)
QUESTIONS = [
    ("How much did 3M spend on property, plant and equipment in 2018 (USD millions)?",
     None, None),
    ("Is Boeing involved in any significant lawsuits?", "BA", None),
    ("What drove the change in gross margin, gross margin, gross margin?", "AMD", 2022),
]


@pytest.mark.parametrize("question,ticker,fy", QUESTIONS)
def test_sql_bm25_matches_python(conn, question, ticker, fy):
    hits = keyword.search(conn, question, k=20, ticker=ticker, fiscal_year=fy)
    ref = bm25_reference(conn, question, ticker, fy)
    assert len(hits) == 20
    assert [round(h.score, 6) for h in hits] == [round(s, 6) for _, s in ref[:20]]
    cutoff = hits[-1].score
    assert ({h.id for h in hits if h.score > cutoff + 1e-6}
            == {cid for cid, s in ref if s > cutoff + 1e-6})


def test_filtered_ranking_is_the_unfiltered_one_restricted(conn):
    question = QUESTIONS[0][0]
    everything = {h.id: h.score for h in keyword.search(conn, question, k=20000)}
    filtered = keyword.search(conn, question, k=30, ticker="MMM", fiscal_year=2018)
    assert all(h.ticker == "MMM" and h.fiscal_year == 2018 for h in filtered)
    assert [h.score for h in filtered] == [everything[h.id] for h in filtered]


def test_keyword_index_lengths_and_odd_questions(conn):
    # dl is the sum of the chunk's term frequencies
    bad = conn.execute("""SELECT count(*) FROM keyword_docs d
                          LEFT JOIN (SELECT id, sum(tf) AS s FROM keyword_postings GROUP BY id) p
                          USING (id) WHERE d.dl <> coalesce(p.s, 0)""").fetchone()[0]
    assert bad == 0
    assert keyword.search(conn, "the of and to") == []  # stop words only
    hits = keyword.search(conn, "What's O'Neil's \\ (non-GAAP) 'margin' | & ! <-> :*?", k=3)
    assert len(hits) == 3
