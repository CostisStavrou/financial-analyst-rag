import random

import pytest

from finrag.eval.retrieval import (
    SETTINGS,
    Passage,
    Question,
    check_index,
    format_report,
    gold_passage,
    min_shared,
    normalise,
    report,
    score,
    summarise,
    windows,
)
from finrag.ingest.chunk import Chunk
from finrag.retrieve.vector import Hit

# 360 random letters: every 60-character window is unique, as in real filings.
rng = random.Random(0)
EVIDENCE = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(360))


def chunk(id: str, page: int, content: str, fy: int = 2022, document: str = "10-K",
          page_end: int | None = None, accession: str = "a") -> Chunk:
    return Chunk(id=id, accession=accession, ticker="ACME", company="Acme", fiscal_year=fy,
                 document=document, item="8", section="Item 8.", page_start=page,
                 page_end=page_end or page, page_label=None, chunk_type="text",
                 content=content, n_tokens=1, source_url="u")


def hit(c: Chunk) -> Hit:
    return Hit(c.id, c.ticker, c.company, c.fiscal_year, c.document, c.section, c.page_start,
               c.page_end, c.page_label, c.chunk_type, c.content, c.source_url, 0.5)


def test_windows_cover_the_whole_text():
    assert normalise("Net sales: $34,229 (2022)") == "netsales342292022"
    assert windows("") == []
    assert windows("a" * 59) == ["a" * 59]  # shorter than a window: the text itself
    text = "".join(chr(97 + i % 26) for i in range(100))
    ws = windows(text)
    assert [len(w) for w in ws] == [60, 60, 60]
    assert ws[0] == text[:60] and ws[1] == text[30:90] and ws[-1] == text[40:]  # ends at end
    assert len(windows(EVIDENCE)) == 11  # starts 0, 30, ..., 300
    assert [min_shared(n) for n in (1, 10, 11, 21, 100)] == [1, 1, 2, 3, 3]


def test_gold_passage():
    chunks = [
        chunk("first-part", 5, EVIDENCE[:240]),  # 7 of 11 windows
        chunk("rest-same-page", 5, "Total " + EVIDENCE[240:]),  # 3 windows, same page
        chunk("shared-header", 9, EVIDENCE[:90]),  # 2 windows, another page
        chunk("full-repeat", 20, EVIDENCE.upper()),  # all 11 windows (case ignored)
        chunk("unrelated", 30, "nothing in common " * 20),
    ]
    p = gold_passage(EVIDENCE, [(normalise(c.content), c) for c in chunks])
    assert p.chunk_ids == {"first-part", "rest-same-page", "full-repeat"}
    assert p.pages == {("ACME", 2022, "10-K", 5), ("ACME", 2022, "10-K", 20)}
    # without the full repeat, the first part (more than half) is the core chunk
    p = gold_passage(EVIDENCE, [(normalise(c.content), c) for c in chunks[:3]])
    assert p.chunk_ids == {"first-part", "rest-same-page"}
    assert gold_passage(EVIDENCE, [(normalise(chunks[4].content), chunks[4])]) is None


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params):
        return self.rows


def db_rows(chunks: list[Chunk], embedded=lambda c: True) -> list[tuple]:
    """Rows as check_index selects them from the chunks table."""
    return [(c.id, c.content_sha, c.ticker, c.fiscal_year, c.document, c.page_start,
             c.page_end, c.accession, embedded(c)) for c in chunks]


def test_check_index_spots_a_stale_database():
    files = [chunk("c0", 1, "alpha"), chunk("c1", 2, "beta")]
    check_index(FakeConn(db_rows(files)), files, "m")
    # a re-parse inserted a chunk: c1 now holds what used to be c0's neighbour
    reparsed = [chunk("c0", 1, "alpha"), chunk("c1", 2, "new"), chunk("c2", 3, "beta")]
    with pytest.raises(RuntimeError, match="missing from the database.*changed since"):
        check_index(FakeConn(db_rows(files)), reparsed, "m")
    # same text and first page, but the chunk now ends a page later: hits are scored on pages
    longer = [chunk("c0", 1, "alpha"), chunk("c1", 2, "beta", page_end=3)]
    with pytest.raises(RuntimeError, match="1 chunks changed since"):
        check_index(FakeConn(db_rows(files)), longer, "m")
    with pytest.raises(RuntimeError, match="without a m embedding"):
        check_index(FakeConn(db_rows(files, lambda c: c.id == "c0")), files, "m")
    # a filing whose chunk file is gone: `index` won't prune it
    dropped = chunk("old-0", 1, "gamma", accession="b")
    with pytest.raises(RuntimeError, match="no chunk file.*db-init --reset") as e:
        check_index(FakeConn(db_rows(files + [dropped])), files, "m")
    assert "Run `finrag index`" not in str(e.value)


def test_short_passage_is_not_placed():
    # "Net income" appears all over a filing: too short to say which page it came from
    chunks = [chunk(f"c{i}", i, "Net income | 5,777 | 6,123") for i in range(30)]
    assert gold_passage("Net income", [(normalise(c.content), c) for c in chunks]) is None


def test_score_ranks_and_other_filings():
    p5, p9 = chunk("p5", 5, ""), chunk("p9", 9, "")
    q = Question("q1", "ACME", 2022, "?", "novel-generated",
                 passages=[Passage({"p5"}, {("ACME", 2022, "10-K", 5)}),
                           Passage({"p9"}, {("ACME", 2022, "10-K", 9)})],
                 not_found=0)
    same_page_other_year = chunk("p5-2021", 5, "", fy=2021)
    same_page_exhibit = chunk("p5-ex13", 5, "", document="EX-13.1")  # CVS FY2018 has both
    neighbour = chunk("p9-other-chunk", 9, "")
    hits = [hit(c) for c in (same_page_other_year, same_page_exhibit, neighbour,
                             chunk("x", 30, ""), p5, p9)]
    s = score(q, hits)
    assert s["page_rank"] == 3  # page 9 of the right document and filing
    assert s["chunk_rank"] == 5
    assert s["passage_ranks"] == [5, 3]
    assert s["all_rank"] == 5
    assert score(q, hits[:4])["all_rank"] is None


def test_passage_not_found_counts_as_missed():
    q = Question("q2", "ACME", 2022, "?", "metrics-generated",
                 passages=[Passage({"p5"}, {("ACME", 2022, "10-K", 5)})], not_found=1)
    s = score(q, [hit(chunk("p5", 5, ""))])
    assert s["page_rank"] == 1
    assert s["passage_ranks"] == [1, None] and s["all_rank"] is None
    m = summarise([s], k=20)
    assert m["multi_passage_questions"] == 1 and m["full_recall@20"] == 0


def test_summarise_maths():
    scores = [
        {"page_rank": 1, "chunk_rank": 2, "passage_ranks": [1], "all_rank": 1},
        {"page_rank": 3, "chunk_rank": None, "passage_ranks": [3, 8], "all_rank": 8},
        {"page_rank": None, "chunk_rank": None, "passage_ranks": [None, None], "all_rank": None},
    ]
    m = summarise(scores, k=20)
    assert m["questions"] == 3
    assert (m["page_hit@1"], m["page_hit@3"], m["page_hit@20"]) == (0.3333, 0.6667, 0.6667)
    assert m["chunk_hit@1"] == 0 and m["chunk_hit@3"] == 0.3333
    assert m["page_mrr"] == round((1 + 1 / 3) / 3, 4)
    assert m["chunk_mrr"] == round(0.5 / 3, 4)
    assert m["multi_passage_questions"] == 2
    assert (m["full_recall@5"], m["full_recall@10"]) == (0, 0.5)
    assert "page_hit@10" not in summarise(scores, k=5)
    assert summarise([], k=20) == {"questions": 0}


def test_report_without_multi_passage_questions():
    s = {"page_rank": 2, "chunk_rank": None, "passage_ranks": [2], "all_rank": 2,
         "search_ms": 5.0, "used_hnsw": True}
    rows = [{"question_type": "novel-generated", "ticker": "ACME", "embed_ms": 300.0,
             "embed_cached": cached,
             **{setting: {**s, "used_hnsw": hnsw} for setting in SETTINGS}}
            for cached, hnsw in ((False, True), (True, False))]
    summary = report(rows, k=20)
    assert summary["unfiltered"]["full_recall@5"] is None
    assert summary["company_year"]["searches_using_hnsw"] == 1
    assert summary["query_embedding"]["api_calls"] == 1
    text = format_report(summary, k=20)
    assert "–" in text and "1/2" in text
