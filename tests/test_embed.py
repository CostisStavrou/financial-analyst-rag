import numpy as np
import pytest

from finrag.index import embed
from finrag.index.embed import Embedder, EmbeddingCache, GeminiEmbedder, RateLimited


def make(monkeypatch, fail_first: int = 0, rpm: int = 0) -> tuple[GeminiEmbedder, list]:
    """GeminiEmbedder with a fake API call: no network, no key."""
    e = GeminiEmbedder.__new__(GeminiEmbedder)
    e.model, e.dim = "fake", 4
    e.tokens_sent = e.requests = e.rate_limited = 0
    e.rpm, e.tpm = rpm, 0
    e.max_batch, e.max_batch_tokens = (min(100, rpm) if rpm else 100), 20_000
    e._sent = embed.deque()
    calls: list[int] = []

    def fake_call(texts, task):
        calls.append(len(texts))
        if len(calls) <= fail_first:
            raise RateLimited(0.0, "EmbedContentRequestsPerMinute-FreeTier")
        return [np.full(4, float(t)) for t in texts]

    e._call = fake_call
    monkeypatch.setattr(embed.time, "sleep", lambda s: None)
    return e, calls


def test_backoff_halves_batch_and_keeps_order(monkeypatch):
    e, calls = make(monkeypatch, fail_first=2)
    texts = [str(i) for i in range(30)]
    out = e.embed(texts, "RETRIEVAL_DOCUMENT")
    assert [int(v[0]) for v in out] == list(range(30))
    assert calls[:3] == [30, 15, 7]  # failed, failed (halved each time), then succeeds
    assert e.rate_limited == 2


def test_pacing_caps_batch_and_waits(monkeypatch):
    e, calls = make(monkeypatch, rpm=10)
    clock = iter(range(0, 10_000, 1))
    monkeypatch.setattr(embed.time, "monotonic", lambda: float(next(clock)))
    out = e.embed([str(i) for i in range(25)], "RETRIEVAL_DOCUMENT")
    assert len(out) == 25
    assert calls == [10, 10, 5]  # never more than 10 texts per request


def test_financebench_filings_first():
    from finrag.config import Company
    from finrag.eval.financebench import question_filings

    cos = [Company("3M", "MMM", 1, "3M", (2018, 2022)), Company("AMD", "AMD", 2, "AMD", (2022,))]
    rows = [
        {"company": "AMD", "doc_type": "10k", "doc_period": 2022},
        {"company": "AMD", "doc_type": "10k", "doc_period": 2022},
        {"company": "3M", "doc_type": "10k", "doc_period": 2018},
        {"company": "3M", "doc_type": "10k", "doc_period": 2015},  # year not in corpus
        {"company": "3M", "doc_type": "10q", "doc_period": 2022},  # not a 10-K
        {"company": "Nike", "doc_type": "10k", "doc_period": 2022},  # company not in corpus
    ]
    assert question_filings(rows, cos) == [("AMD", 2022), ("MMM", 2018)]


def test_an_offline_embedder_reads_the_cache_and_never_calls_the_api(tmp_path):
    e = Embedder("gemini", "m", 3, api_key="", cache_dir=tmp_path, offline=True)
    v = np.array([1, 0, 0], dtype=np.float32)
    e.cache.put_many({EmbeddingCache.key("m", 3, "RETRIEVAL_QUERY", "cached"): v})
    assert (e.embed_query("cached") == v).all()
    with pytest.raises(RuntimeError, match="offline"):
        e.embed_query("not cached")
