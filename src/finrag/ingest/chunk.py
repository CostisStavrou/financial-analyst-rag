"""Split parsed blocks into retrieval chunks that keep company, fiscal year, section and page.

- Text: consecutive paragraphs of one section packed to ~TARGET_TOKENS, never crossing a
  section boundary; a short last paragraph is repeated at the start of the next chunk.
- Tables: one chunk per table, with a short preceding paragraph as its caption. Very long
  tables are split by rows, repeating the header row.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from finrag.ingest.parse_html import Block, ParsedDoc

TARGET_TOKENS = 500
OVERLAP_MAX_TOKENS = 100
TABLE_MAX_TOKENS = 1000
CAPTION_MAX_TOKENS = 60
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def est_tokens(text: str) -> int:
    """Rough token count (~4 characters per token), enough for sizing chunks."""
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    id: str
    accession: str
    ticker: str
    company: str
    fiscal_year: int
    document: str  # "10-K" or the exhibit type, e.g. "EX-13"
    item: str
    section: str
    page_start: int
    page_end: int
    page_label: str | None
    chunk_type: str  # "text" | "table"
    content: str
    n_tokens: int
    source_url: str

    @property
    def content_sha(self) -> str:
        """Hash of exactly what gets embedded: a change means the vector is stale."""
        return hashlib.sha256(self.embed_text().encode()).hexdigest()

    def embed_text(self) -> str:
        """Content with a short header, so the vector knows which filing it came from."""
        doc = "10-K" if self.document == "10-K" else f"10-K {self.document}"
        header = (
            f"{self.company} ({self.ticker}) {doc}, fiscal year {self.fiscal_year}. "
            f"{self.section}. Page {self.page_start}."
        )
        return f"{header}\n\n{self.content}"

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long_text(text: str, limit: int) -> list[str]:
    parts, cur = [], ""
    for sent in _SENT.split(text):
        if cur and est_tokens(cur) + est_tokens(sent) > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    # A "sentence" longer than the limit (rare: run-on lists) is cut by characters.
    out = []
    for p in parts:
        while est_tokens(p) > limit * 1.5:
            out.append(p[: limit * 4])
            p = p[limit * 4 :]
        out.append(p)
    return out


def _split_table(text: str, limit: int) -> list[str]:
    rows = text.split("\n")
    if est_tokens(text) <= limit or len(rows) < 3:
        return [text]
    header, parts, cur = rows[0], [], [rows[0]]
    for row in rows[1:]:
        if len(cur) > 1 and est_tokens("\n".join(cur + [row])) > limit:
            parts.append("\n".join(cur))
            cur = [header]
        cur.append(row)
    if len(cur) > 1:
        parts.append("\n".join(cur))
    return parts


def chunk_document(
    doc: ParsedDoc,
    *,
    accession: str,
    ticker: str,
    company: str,
    fiscal_year: int,
    document: str,
    source_url: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    doc_tag = "10K" if document == "10-K" else document.replace("-", "").replace(".", "")

    def emit(blocks: list[Block], kind: str, content: str | None = None) -> None:
        content = content or "\n\n".join(b.text for b in blocks)
        first, last = blocks[0], blocks[-1]
        chunks.append(
            Chunk(
                id=f"{ticker}-FY{fiscal_year}-{doc_tag}-{len(chunks):04d}",
                accession=accession,
                ticker=ticker,
                company=company,
                fiscal_year=fiscal_year,
                document=document,
                item=first.item,
                section=first.section,
                page_start=first.page,
                page_end=last.page,
                page_label=doc.page_labels.get(first.page),
                chunk_type=kind,
                content=content,
                n_tokens=est_tokens(content),
                source_url=source_url,
            )
        )

    buf: list[Block] = []

    def flush(carry: bool = False) -> None:
        nonlocal buf
        if buf:
            emit(buf, "text")
        tail = buf[-1] if buf else None
        buf = [tail] if carry and tail and est_tokens(tail.text) <= OVERLAP_MAX_TOKENS else []

    for block in doc.blocks:
        if buf and block.section != buf[-1].section:
            flush()
        if block.kind == "table":
            caption = None
            if buf and est_tokens(buf[-1].text) <= CAPTION_MAX_TOKENS:
                caption = buf.pop()
            flush()
            for part in _split_table(block.text, TABLE_MAX_TOKENS):
                blocks = [caption, block] if caption else [block]
                content = f"{caption.text}\n\n{part}" if caption else part
                emit(blocks, "table", content)
            continue
        pieces = (
            _split_long_text(block.text, TARGET_TOKENS)
            if est_tokens(block.text) > TARGET_TOKENS
            else [block.text]
        )
        for text in pieces:
            piece = Block(block.kind, text, block.page, block.item, block.section)
            size = sum(est_tokens(b.text) for b in buf)
            if buf and size + est_tokens(text) > TARGET_TOKENS:
                flush(carry=True)
            buf.append(piece)
    flush()
    return chunks
