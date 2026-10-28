"""Hybrid retrieval: vector and BM25 rankings merged with Reciprocal Rank Fusion.

RRF (Cormack, Clarke & Büttcher, SIGIR 2009): a chunk scores the sum over the two legs of
1 / (RRF_K + its rank in that leg's top DEPTH), with equal weights. Only ranks are used, so
cosine and BM25 scales never meet.

With RRF_K 60 and DEPTH 50, a chunk in both legs' lists always beats a chunk in only one
(2 / (60 + 50) > 1 / (60 + 1)): the fused ranking puts the legs' agreement first.

Ties: equal scores within a leg share a rank. Equal fused scores (a vector-only and a
keyword-only chunk at the same rank, say) go to the better vector rank, then the better
keyword rank, then md5(id).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import psycopg

from finrag.retrieve import keyword, vector
from finrag.retrieve.vector import Hit

RRF_K = 60
DEPTH = 50  # chunks taken from each leg; at most vector.EF_SEARCH


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def competition_ranks(scores: list[float]) -> list[int]:
    """1-based ranks of scores sorted high to low; equal scores share a rank (1, 2, 2, 4)."""
    ranks: list[int] = []
    for i, s in enumerate(scores):
        ranks.append(ranks[-1] if i and s == scores[i - 1] else i + 1)
    return ranks


@dataclass
class Fused:
    id: str
    score: float
    ranks: dict[str, int | None]  # rank in each leg, None if not in its list


def rrf(legs: dict[str, list[tuple[str, int]]], k: int = RRF_K) -> list[Fused]:
    """Fuse ranked lists of (chunk id, rank). On equal scores, legs earlier in `legs` win."""
    total: dict[str, Fraction] = defaultdict(Fraction)  # exact, so equal sums are equal
    ranks: dict[str, dict[str, int | None]] = {}
    for leg, ranked in legs.items():
        for cid, r in ranked:
            total[cid] += Fraction(1, k + r)
            ranks.setdefault(cid, dict.fromkeys(legs))[leg] = r

    def order(cid: str) -> tuple:
        return (-total[cid], *(ranks[cid][leg] or math.inf for leg in legs), md5(cid))

    return [Fused(cid, float(total[cid]), ranks[cid]) for cid in sorted(total, key=order)]


def fuse(legs: dict[str, list[Hit]], k: int = RRF_K) -> list[Fused]:
    """RRF over legs of hits, each sorted best first."""
    return rrf({leg: list(zip([h.id for h in hits], competition_ranks([h.score for h in hits])))
                for leg, hits in legs.items()}, k)


@dataclass
class HybridResult:
    hits: list[Hit]  # fused top k; score = RRF score
    ranks: list[dict[str, int | None]]  # each hit's rank in the vector and keyword legs
    vector: list[Hit]  # each leg's top DEPTH
    keyword: list[Hit]
    ms: dict[str, float]  # search time per leg


def search(
    conn: psycopg.Connection,
    question: str,
    query_vec: np.ndarray,
    model: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    item: str | None = None,
    exact: bool = False,
    depth: int = DEPTH,
) -> HybridResult:
    """Top k chunks by RRF over the vector and keyword legs (run one after the other).
    `exact` applies to the vector leg."""
    f = {"ticker": ticker, "fiscal_year": fiscal_year, "item": item}
    t0 = time.perf_counter()
    vec = vector.search(conn, query_vec, model, k=depth, exact=exact, **f)
    t1 = time.perf_counter()
    kw = keyword.search(conn, question, k=depth, **f)
    t2 = time.perf_counter()
    fused = fuse({"vector": vec, "keyword": kw})[:k]
    by_id = {h.id: h for h in kw + vec}
    return HybridResult(
        hits=[replace(by_id[x.id], score=x.score) for x in fused],
        ranks=[x.ranks for x in fused],
        vector=vec,
        keyword=kw,
        ms={"vector": 1000 * (t1 - t0), "keyword": 1000 * (t2 - t1)},
    )
