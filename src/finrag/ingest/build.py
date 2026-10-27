"""Parse every downloaded filing into chunks, written as JSONL under data/parsed/."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from finrag.ingest.chunk import Chunk, chunk_document
from finrag.ingest.edgar import iter_downloaded
from finrag.ingest.parse_html import parse_file

log = logging.getLogger(__name__)


def build_chunks(raw_dir: Path, parsed_dir: Path, tickers: set[str] | None = None) -> list[dict]:
    """Returns one summary row per filing."""
    summary = []
    for folder, meta in iter_downloaded(raw_dir):
        if tickers and meta.ticker not in tickers:
            continue
        chunks: list[Chunk] = []
        sections: dict[str, list] = {}
        pages = {}
        for d in meta.documents:
            doc = parse_file(folder / d.filename, d.type)
            doc_chunks = chunk_document(
                doc,
                accession=meta.accession,
                ticker=meta.ticker,
                company=meta.company,
                fiscal_year=meta.fiscal_year,
                document=d.type,
                source_url=d.url,
            )
            # ids carry the document type, so they stay unique across a filing
            chunks.extend(doc_chunks)
            sections[d.type] = doc.sections
            pages[d.type] = doc.n_pages
        out = parsed_dir / meta.ticker / f"FY{meta.fiscal_year}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        kinds = Counter(c.chunk_type for c in chunks)
        row = {
            "ticker": meta.ticker,
            "fiscal_year": meta.fiscal_year,
            "accession": meta.accession,
            "documents": [d.type for d in meta.documents],
            "pages": pages,
            "chunks": len(chunks),
            "text_chunks": kinds["text"],
            "table_chunks": kinds["table"],
            "tokens": sum(c.n_tokens for c in chunks),
            "sections": {k: [s for s, _ in v] for k, v in sections.items()},
        }
        (out.with_suffix(".summary.json")).write_text(json.dumps(row, indent=2))
        summary.append(row)
        log.info(
            "%s FY%s: %s pages, %d chunks (%d tables), sections %s",
            meta.ticker, meta.fiscal_year, pages, len(chunks), kinds["table"],
            row["sections"].get("10-K"),
        )
    return summary


def load_chunks(parsed_dir: Path, tickers: set[str] | None = None) -> list[Chunk]:
    chunks = []
    for path in sorted(parsed_dir.glob("*/FY*.jsonl")):
        if tickers and path.parent.name not in tickers:
            continue
        with path.open() as f:
            chunks.extend(Chunk(**json.loads(line)) for line in f)
    return chunks
