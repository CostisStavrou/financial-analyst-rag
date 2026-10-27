"""Turn an EDGAR 10-K HTML document into ordered blocks (paragraphs and tables) that keep
their page number and 10-K section ("Item 7. Management's Discussion and Analysis ...").

Pages: EDGAR HTML has no pages as such, but filings mark the printed page breaks with CSS
(`page-break-before/after: always`), or with <hr> in some older filings. Page N here is the
Nth such page of the document (1-based). The printed page number in the footer, when there
is one, is kept separately as `page_label` and dropped from the text.

Tables stay whole: each <table> becomes one block rendered as pipe-separated rows. Proper
structured table extraction comes later.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from lxml import html as lxml_html

log = logging.getLogger(__name__)

BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "section", "article",
    "header", "footer", "blockquote", "center", "pre", "dl", "dt", "dd", "body", "html",
    "form",
}
SKIP_TAGS = {"script", "style", "head", "title", "noscript", "ix:header", "xml"}
_WS = re.compile(r"\s+")
_PAGE_LABEL = re.compile(r"^(?:page\s+)?(?:[a-z]{1,2}\s?-\s?)?\d{1,3}$", re.IGNORECASE)
_NAV = re.compile(r"^(table of contents|index|back to contents|return to table of contents)$", re.IGNORECASE)
ITEM_RE = re.compile(r"^items?\s*(\d{1,2})\s*([abc])?\b[\s.:\-–—]*(.*)$", re.IGNORECASE)

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity and Related Stockholder Matters",
    "6": "Selected Financial Data / Reserved",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
FRONT_MATTER = ("0", "Cover page and front matter")


def section_name(item: str) -> str:
    return f"Item {item}. {ITEM_TITLES.get(item, '')}".strip()


@dataclass
class Block:
    kind: str  # "text" | "table"
    text: str
    page: int
    item: str = FRONT_MATTER[0]
    section: str = FRONT_MATTER[1]


@dataclass
class ParsedDoc:
    blocks: list[Block]
    n_pages: int
    page_labels: dict[int, str] = field(default_factory=dict)
    sections: list[tuple[str, int]] = field(default_factory=list)  # (item, first page)


def _clean(text: str) -> str:
    return _WS.sub(" ", text.replace("\xa0", " ").replace("\u200b", "")).strip()


def _style(el) -> str:
    return (el.get("style") or "").lower().replace(" ", "")


def _hidden(el) -> bool:
    return "display:none" in _style(el)


def _breaks(el, hr_breaks: bool) -> tuple[bool, bool]:
    s = _style(el)
    before = "page-break-before:always" in s or "break-before:page" in s
    after = "page-break-after:always" in s or "break-after:page" in s
    if hr_breaks and el.tag == "hr":
        after = True
    return before, after


def _merge_cells(cells: list[str]) -> list[str]:
    """Drop empty cells; glue "$" / "(" to the number after and ")" / "%" to the one before."""
    merged: list[str] = []
    for c in cells:
        if not c:
            continue
        if merged and (merged[-1] in ("$", "(", "$(", "€", "£") or c in (")", "%", ")%", "%)")):
            merged[-1] = (merged[-1] + c).replace("$ ", "$")
        else:
            merged.append(c)
    return merged


def _table_rows(table) -> list[list[str]]:
    rows = []
    for tr in table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"):
        if _hidden(tr):
            continue
        cells = [_clean(td.text_content()) for td in tr.xpath("./td|./th") if not _hidden(td)]
        merged = _merge_cells(cells)
        if merged:
            rows.append(merged)
    return rows


# Some filings (e.g. General Mills FY2022) are PDF-to-HTML conversions: every printed line,
# or table cell, is an absolutely positioned <div> with no <table> tags. For those, rebuild
# lines from the coordinates, then paragraphs (by vertical gap) and tables (lines with
# several cells).
_LEFT = re.compile(r"(?:^|;)left:(-?[\d.]+)px")
_TOP = re.compile(r"(?:^|;)top:(-?[\d.]+)px")


def _layout_page(frags: list[tuple[float, float, str]]) -> list[tuple[str, object]]:
    if not frags:
        return []
    lines: list[tuple[float, list[tuple[float, str]]]] = []
    for top, left, text in sorted(frags):
        if lines and top - lines[-1][0] <= 3:
            lines[-1][1].append((left, text))
        else:
            lines.append((top, [(left, text)]))
    cells = [[t for _, t in sorted(parts)] for _, parts in lines]
    tops = [t for t, _ in lines]
    gaps = sorted(b - a for a, b in pairwise(tops))
    line_gap = gaps[len(gaps) // 2] if gaps else 15.0
    is_row = [len(c) >= 2 for c in cells]
    for i in range(1, len(cells) - 1):  # label-only rows inside a table ("Current assets:")
        if not is_row[i] and is_row[i - 1] and is_row[i + 1]:
            is_row[i] = True

    events: list[tuple[str, object]] = []
    para: list[str] = []
    table: list[list[str]] = []

    def end_para():
        if para:
            events.append(("text", " ".join(para)))
            para.clear()

    def end_table():
        if table:
            events.append(("table", [list(r) for r in table]))
            table.clear()

    for i, row in enumerate(cells):
        gap = tops[i] - tops[i - 1] if i else 0
        if is_row[i]:
            end_para()
            if table and gap > 3 * line_gap:
                end_table()
            merged = _merge_cells(row)
            if merged:
                table.append(merged)
        else:
            end_table()
            text = row[0]
            if ITEM_RE.match(text) and len(text) < 150:  # an item heading is its own block
                end_para()
                events.append(("text", text))
                continue
            if para and gap > 1.45 * line_gap:
                end_para()
            para.append(text)
    end_para()
    end_table()
    return events


def _positioned_events(root) -> list[tuple[str, object]]:
    events: list[tuple[str, object]] = []
    frags: list[tuple[float, float, str]] = []

    def new_page():
        events.extend(_layout_page(frags))
        frags.clear()
        events.append(("page", None))

    def visit(el):
        if not isinstance(el.tag, str) or el.tag.lower() in SKIP_TAGS or _hidden(el):
            return
        before, after = _breaks(el, False)
        if before:
            new_page()
        style = _style(el)
        if "position:absolute" in style:
            text = _clean(el.text_content())
            left, top = _LEFT.search(style), _TOP.search(style)
            if text and left and top:
                frags.append((float(top.group(1)), float(left.group(1)), text))
        else:
            for child in el:
                visit(child)
        if after:
            new_page()

    visit(root)
    events.extend(_layout_page(frags))
    return events


class _Walker:
    def __init__(self, hr_breaks: bool):
        self.hr_breaks = hr_breaks
        self.events: list[tuple[str, object]] = []  # ("text", str) | ("table", rows) | ("page", None)
        self.buf: list[str] = []

    def flush(self):
        text = _clean(" ".join(self.buf))
        self.buf = []
        if text:
            self.events.append(("text", text))

    def page(self):
        self.flush()
        self.events.append(("page", None))

    def walk(self, el):
        tag = el.tag if isinstance(el.tag, str) else None
        if tag is None or tag.lower() in SKIP_TAGS or _hidden(el):
            if el.tail:
                self.buf.append(el.tail)
            return
        tag = tag.lower()
        before, after = _breaks(el, self.hr_breaks)
        if before:
            self.page()
        if tag == "table":
            self.flush()
            rows = _table_rows(el)
            if rows:
                self.events.append(("table", rows))
        elif tag in ("br", "hr"):
            self.flush()
        elif tag in BLOCK_TAGS:
            self.flush()
            if el.text:
                self.buf.append(el.text)
            for child in el:
                self.walk(child)
            self.flush()
        else:
            if el.text:
                self.buf.append(el.text)
            for child in el:
                self.walk(child)
        if after:
            self.page()
        if el.tail:
            self.buf.append(el.tail)


def _layout_table_to_text(rows: list[list[str]]) -> list[str] | None:
    """Tables used for layout (headings, bullet lists, single lines) become paragraphs."""
    if len(rows) == 1:
        return [" ".join(rows[0])]
    if all(len(r) == 1 for r in rows):
        return [r[0] for r in rows]
    if all(len(r) == 2 and len(r[0]) <= 3 for r in rows):  # bullet / (1) + text
        return [" ".join(r) for r in rows]
    return None


def _render_table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def parse_html(content: bytes) -> ParsedDoc:
    root = lxml_html.document_fromstring(content)
    css_breaks = len(root.xpath('//*[contains(translate(@style," ",""),"page-break")'
                                ' or contains(translate(@style," ",""),"break-before:page")'
                                ' or contains(translate(@style," ",""),"break-after:page")]'))
    if len(root.xpath('//div[contains(translate(@style," ",""),"position:absolute")]')) > 2000:
        events = _positioned_events(root)
    else:
        walker = _Walker(hr_breaks=css_breaks < 5)
        walker.walk(root)
        walker.flush()
        events = walker.events

    blocks: list[Block] = []
    page_labels: dict[int, str] = {}
    page, page_start = 1, 0
    for kind, payload in events + [("page", None)]:
        if kind == "page":
            if len(blocks) == page_start:
                continue  # empty page (double break)
            last = blocks[-1]
            if last.kind == "text" and _PAGE_LABEL.match(last.text):
                page_labels[page] = last.text
                blocks.pop()
            page += 1
            page_start = len(blocks)
        elif kind == "text":
            if not _NAV.match(payload):
                blocks.append(Block("text", payload, page))
        else:
            as_text = _layout_table_to_text(payload)
            if as_text is not None:
                blocks.extend(Block("text", t, page) for t in as_text if not _NAV.match(t))
            else:
                blocks.append(Block("table", _render_table(payload), page))
    n_pages = max((b.page for b in blocks), default=0)
    doc = ParsedDoc(blocks=blocks, n_pages=n_pages, page_labels=page_labels)
    assign_sections(doc)
    return doc


def _item_key(item: str) -> tuple[int, str]:
    return int(re.match(r"\d+", item).group()), item[len(re.match(r"\d+", item).group()):]


def _looks_like_toc(run: list[tuple[int, str]], n_blocks: int) -> bool:
    """A run of back-to-back item headings is a contents list if it sits near the start or
    spans early and late items. Short real sections in a row (Items 10-14 "incorporated by
    reference to the proxy statement") are neither."""
    keys = [_item_key(item) for _, item in run]
    early = run[0][0] < 0.15 * n_blocks
    wide = min(keys)[0] <= 4 and max(keys)[0] >= 7
    return early or wide


def assign_sections(doc: ParsedDoc) -> None:
    """Label each block with its 10-K item.

    Item headings are short paragraphs starting "Item 7." etc. The table of contents (and
    cross-reference lists) show up as runs of item headings with almost no text between
    them; those runs are dropped. Remaining headings must come in non-decreasing order.
    """
    blocks = doc.blocks
    cands: list[tuple[int, str]] = []
    for i, b in enumerate(blocks):
        # Combined headings ("Items 10, 11, 12 and 13. Directors, ...; Executive ...") run long.
        if b.kind != "text" or len(b.text) > (300 if b.text[:5].lower() == "items" else 150):
            continue
        m = ITEM_RE.match(b.text)
        if m and 1 <= int(m.group(1)) <= 16:
            item = m.group(1) + (m.group(2) or "").upper()
            if item in ITEM_TITLES:
                cands.append((i, item))

    # Characters of content between each candidate and the next.
    spans = []
    for k, (i, _) in enumerate(cands):
        end = cands[k + 1][0] if k + 1 < len(cands) else len(blocks)
        spans.append(sum(len(b.text) for b in blocks[i + 1 : end]))
    toc = set()
    run: list[int] = []
    for k, span in enumerate(spans):
        # The entry that ends a run (followed by real text) is the contents list's last line
        # if it carries on the ascending sequence, or the first real heading if the
        # numbering restarts (contents list directly followed by "Item 1. Business").
        if span >= 400 or k + 1 == len(spans):
            if not run or _item_key(cands[k][1]) > _item_key(cands[run[-1]][1]):
                run.append(k)
        else:
            run.append(k)
            continue
        if len(run) >= 4 and _looks_like_toc([cands[j] for j in run], len(blocks)):
            toc.update(run)
        run = []
    heads: list[tuple[int, str]] = []
    for k, (i, item) in enumerate(cands):
        if k in toc:
            continue
        if heads and _item_key(item) < _item_key(heads[-1][1]):
            continue
        if heads and item == heads[-1][1]:
            continue  # repeated heading (e.g. "Item 8" then "Item 8 (continued)")
        heads.append((i, item))

    # Some filings split "Item 7." and its title into two paragraphs: fine, we use the
    # canonical title. Apply labels.
    starts = dict(heads)
    item, section = FRONT_MATTER
    for i, b in enumerate(blocks):
        if i in starts:
            item = starts[i]
            section = section_name(item)
            doc.sections.append((item, b.page))
        b.item, b.section = item, section


# Exhibit 13 (annual report to shareholders) has no "Item" headings; map its main parts
# to the 10-K items they stand in for.
_EX13_HEADINGS = [
    (re.compile(r"management.s discussion and analysis", re.IGNORECASE), "7"),
    (re.compile(r"quantitative and qualitative disclosures? about market risk", re.IGNORECASE), "7A"),
    (re.compile(r"(report of independent registered public accounting firm|"
                r"consolidated statements? of (income|operations|earnings))", re.IGNORECASE), "8"),
]


def assign_exhibit_sections(doc: ParsedDoc, label: str) -> None:
    item, section = "EX", label
    for b in doc.blocks:
        if b.kind == "text" and len(b.text) < 120:
            for rx, it in _EX13_HEADINGS:
                if rx.search(b.text) and (item == "EX" or _item_key(it) >= _item_key(item)):
                    if it != item:
                        item, section = it, f"{section_name(it)} ({label})"
                        doc.sections.append((item, b.page))
                    break
        b.item, b.section = item, section


def parse_file(path: Path, doc_type: str = "10-K") -> ParsedDoc:
    doc = parse_html(path.read_bytes())
    if doc_type.upper().startswith("EX-13"):
        doc.sections = []
        assign_exhibit_sections(doc, "Exhibit 13")
    return doc
