import pytest

from finrag.ingest.chunk import chunk_document
from finrag.ingest.edgar import check_user_agent, fiscal_year_of
from finrag.ingest.parse_html import parse_html

BREAK = '<div style="page-break-after: always"></div>'
FILLER = "Revenue grew because customers bought more widgets in every region. " * 12

HTML = f"""
<html><body>
<div style="display:none"><ix:header>hidden 123456 facts</ix:header></div>
<p>ACME CORP FORM 10-K</p>
<table>
  <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
  <tr><td>Item 7.</td><td>MD&amp;A</td><td>20</td></tr>
  <tr><td>Item 8.</td><td>Financial Statements</td><td>40</td></tr>
</table>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. MD&amp;A</p>
<p>Item 8. Financial Statements</p><p>Item 15. Exhibits</p>
<p>1</p>
{BREAK}
<p><a href="#toc">Table of Contents</a></p>
<p style="font-weight:bold">ITEM 1. BUSINESS</p>
<p>{FILLER}</p>
<p>2</p>
{BREAK}
<p>Item 1A. Risk Factors</p>
<p>{FILLER}</p>
<p>3</p>
{BREAK}
<p>Item 7. Management's Discussion and Analysis</p>
<p>{FILLER}</p>
<p>See Item 8 for the statements.</p>
<p>4</p>
{BREAK}
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>CONSOLIDATED STATEMENT OF INCOME</p>
<table>
  <tr><td></td><td>2022</td><td></td><td>2021</td></tr>
  <tr><td>Net sales</td><td>$</td><td>34,229</td><td>$</td><td>35,355</td></tr>
  <tr><td>Net income</td><td>$</td><td>5,777</td><td>$</td><td>(5,921</td><td>)</td></tr>
</table>
<p>F-1</p>
</body></html>
"""


def test_pages_sections_tables():
    doc = parse_html(HTML.encode())
    text = " ".join(b.text for b in doc.blocks)
    assert "hidden" not in text  # ix:header dropped
    assert "Table of Contents" not in text
    assert doc.n_pages == 5
    assert doc.page_labels == {1: "1", 2: "2", 3: "3", 4: "4", 5: "F-1"}
    assert [s for s, _ in doc.sections] == ["1", "1A", "7", "8"]  # contents list ignored
    assert dict(doc.sections)["8"] == 5
    table = [b for b in doc.blocks if b.kind == "table"]
    assert len(table) == 2  # contents table + income statement
    assert table[0].item == "0" and table[0].page == 1  # contents sit in the front matter
    assert "| Net sales | $34,229 | $35,355 |" in table[-1].text
    assert "$(5,921)" in table[-1].text
    assert table[-1].item == "8" and table[-1].page == 5


def test_chunks_keep_metadata():
    doc = parse_html(HTML.encode())
    chunks = chunk_document(doc, accession="0000000000-23-000001", ticker="ACME",
                            company="Acme", fiscal_year=2022, document="10-K",
                            source_url="https://www.sec.gov/x.htm")
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert tables[-1].content.startswith("CONSOLIDATED STATEMENT OF INCOME")
    assert tables[-1].section.startswith("Item 8.") and tables[-1].page_start == 5
    assert tables[-1].page_label == "F-1"
    risk = [c for c in chunks if c.item == "1A"]
    assert risk and all(c.page_start == 3 for c in risk)
    assert len({c.id for c in chunks}) == len(chunks)
    assert "Acme (ACME) 10-K, fiscal year 2022" in chunks[0].embed_text()


def test_fiscal_year_labels():
    assert fiscal_year_of("2022-12-31") == 2022
    assert fiscal_year_of("2023-01-28") == 2023  # Best Buy fiscal 2023
    assert fiscal_year_of("2023-06-30") == 2023  # Amcor fiscal 2023
    assert fiscal_year_of("2016-01-02") == 2015  # 52/53-week year ending early January


def test_user_agent_required():
    with pytest.raises(ValueError):
        check_user_agent("")
    with pytest.raises(ValueError):
        check_user_agent("someone@example.com")
    with pytest.raises(ValueError):
        check_user_agent("just-a-name")
    assert check_user_agent("Jo Bloggs jo@uni.ac.uk") == "Jo Bloggs jo@uni.ac.uk"
