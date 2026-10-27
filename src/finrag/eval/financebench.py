"""FinanceBench open-source sample (PatronusAI/financebench on Hugging Face, CC BY-NC 4.0).

Downloaded on demand into data/financebench/ (git-ignored), not redistributed in this repo.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import requests

from finrag.config import Company

URL = "https://huggingface.co/datasets/PatronusAI/financebench/resolve/main/financebench_merged.jsonl"


def download(data_dir: Path) -> Path:
    path = data_dir / "financebench" / "financebench_merged.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def coverage(rows: list[dict], companies: list[Company]) -> dict:
    """How many FinanceBench 10-K questions the configured corpus can answer."""
    wanted = {(c.financebench, y) for c in companies for y in c.fiscal_years}
    tenk = [r for r in rows if r["doc_type"].lower() == "10k"]
    in_scope = [r for r in tenk if (r["company"], int(r["doc_period"])) in wanted]
    per_company = Counter(r["company"] for r in in_scope)
    return {
        "questions": len(rows),
        "questions_on_10k": len(tenk),
        "in_scope": len(in_scope),
        "per_company": dict(per_company.most_common()),
        "in_scope_ids": [r["financebench_id"] for r in in_scope],
    }
