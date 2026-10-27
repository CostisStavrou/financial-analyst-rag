"""Check chunk page numbers against FinanceBench's evidence pages.

For each in-scope FinanceBench question, find the chunk that contains its evidence passage
(any 60-character window, ignoring case and punctuation) and compare our page with
FinanceBench's `evidence_page_num` (0-based page index in the company's PDF).
"""

from __future__ import annotations

import ast
import re

from finrag.config import Company
from finrag.ingest.chunk import Chunk

_NORM = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM.sub("", text.lower())


def page_check(rows: list[dict], companies: list[Company], chunks: list[Chunk]) -> dict:
    by_name = {c.financebench: c for c in companies}
    index: dict[tuple[str, int], list[tuple[str, Chunk]]] = {}
    for c in chunks:
        index.setdefault((c.ticker, c.fiscal_year), []).append((_norm(c.content), c))

    passages = found = same = within_one = 0
    misses = []
    for r in rows:
        co = by_name.get(r["company"])
        fy = int(r["doc_period"])
        if r["doc_type"].lower() != "10k" or not co or fy not in co.fiscal_years:
            continue
        evidence = r["evidence"]
        if isinstance(evidence, str):
            evidence = ast.literal_eval(evidence)
        for e in evidence:
            passages += 1
            text = _norm(e["evidence_text"])
            hit = None
            for start in range(0, max(1, len(text) - 60), 40):
                window = text[start : start + 60]
                hit = next((c for n, c in index.get((co.ticker, fy), []) if window in n), None)
                if hit:
                    break
            if not hit:
                misses.append(f"{co.ticker} FY{fy} {r['financebench_id']}: passage not found")
                continue
            found += 1
            diff = hit.page_start - (e["evidence_page_num"] + 1)
            same += diff == 0
            within_one += abs(diff) <= 1
            if abs(diff) > 1:
                misses.append(
                    f"{co.ticker} FY{fy} {r['financebench_id']}: FinanceBench page "
                    f"{e['evidence_page_num'] + 1}, ours {hit.page_start} ({hit.document})"
                )
    return {
        "evidence_passages": passages,
        "found_in_chunks": found,
        "same_page": same,
        "within_1_page": within_one,
        "differences": misses,
    }
