"""SEC EDGAR 10-K downloader.

Official SEC endpoints only:
  https://data.sec.gov/submissions/CIK##########.json             filing history
  https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<acc>-index.htm  documents in a filing
  https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<file>           the documents

SEC fair-access rules: declare a User-Agent with a name and contact email, and send no more
than 10 requests a second. This client waits at least 0.5 s between requests (<= 2/s).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import requests
from lxml import html as lxml_html

from finrag.config import Company

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{file}"
# Exhibit 13 (annual report to shareholders) holds MD&A and financials for companies that
# incorporate them into the 10-K by reference.
WANTED_EXHIBITS = re.compile(r"^EX-13", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def check_user_agent(ua: str) -> str:
    """SEC needs a name and a contact email. Refuse placeholders rather than guess."""
    ua = ua.strip()
    if not _EMAIL.search(ua) or len(_EMAIL.sub("", ua).strip()) < 2:
        raise ValueError(
            'SEC_USER_AGENT must be "Full Name email@domain" (set it in .env). '
            "SEC blocks clients that don't identify themselves."
        )
    if "example.com" in ua.lower():
        raise ValueError("SEC_USER_AGENT still holds the example placeholder.")
    return ua


class SecClient:
    """requests.Session with SEC headers, a request-rate ceiling and retries."""

    def __init__(self, user_agent: str, min_interval: float = 0.5, max_retries: int = 5):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": check_user_agent(user_agent), "Accept-Encoding": "gzip, deflate"}
        )
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0
        self.requests_made = 0

    def get(self, url: str) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests_made += 1
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                backoff = float(resp.headers.get("Retry-After") or 2 ** (attempt + 1))
                log.warning("SEC %s on %s; retrying in %.0fs", resp.status_code, url, backoff)
                time.sleep(backoff)
                continue
            if resp.status_code == 403:
                raise RuntimeError(
                    f"SEC returned 403 for {url}. Check SEC_USER_AGENT and the request rate."
                )
            resp.raise_for_status()
        raise RuntimeError(f"Gave up on {url}")

    def get_json(self, url: str) -> dict:
        return self.get(url).json()


@dataclass
class Filing:
    cik: int
    accession: str  # dashed form, e.g. 0000066740-23-000014
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    fiscal_year: int

    @property
    def folder(self) -> str:
        return self.accession.replace("-", "")


@dataclass
class FilingDoc:
    type: str
    filename: str
    url: str
    sha256: str = ""
    bytes: int = 0


@dataclass
class FilingMeta:
    company: str
    ticker: str
    cik: int
    fiscal_year: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    index_url: str
    documents: list[FilingDoc] = field(default_factory=list)
    downloaded_at: str = ""


def fiscal_year_of(report_date: str) -> int:
    """Company fiscal-year label from the period end date.

    The label is the calendar year the fiscal year ends in (Best Buy FY2023 ended
    28 Jan 2023, Amcor FY2023 ended 30 Jun 2023). A 52/53-week year that spills a few days
    into January belongs to the previous label.
    """
    d = date.fromisoformat(report_date)
    return d.year - 1 if d.month == 1 and d.day <= 7 else d.year


def _rows(block: dict) -> list[dict]:
    keys = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]
    return [dict(zip(keys, vals)) for vals in zip(*(block[k] for k in keys))]


def list_10k_filings(client: SecClient, cik: int, wanted_years: set[int]) -> dict[int, Filing]:
    """Original 10-K per fiscal year (no 10-K/A). Older history pages are fetched only if needed."""
    sub = client.get_json(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
    log.info("CIK %s = %s", cik, sub.get("name"))
    pages = [sub["filings"]["recent"]]
    older = [f["name"] for f in sub["filings"].get("files", [])]
    found: dict[int, Filing] = {}
    while True:
        for row in _rows(pages.pop()):
            if row["form"] != "10-K":
                continue
            if row["reportDate"]:
                fy = fiscal_year_of(row["reportDate"])
            else:  # rare on old filings; 10-Ks are filed after the year they cover
                fy = int(row["filingDate"][:4]) - 1
                log.warning("No reportDate for %s; assuming FY%s", row["accessionNumber"], fy)
            if fy not in wanted_years:
                continue
            filing = Filing(
                cik=cik,
                accession=row["accessionNumber"],
                form=row["form"],
                filing_date=row["filingDate"],
                report_date=row["reportDate"],
                primary_document=row["primaryDocument"],
                fiscal_year=fy,
            )
            # Keep the earliest-filed original if two 10-Ks map to one year.
            if fy not in found or filing.filing_date < found[fy].filing_date:
                found[fy] = filing
        if wanted_years <= found.keys() or not older:
            return found
        pages.append(client.get_json(SUBMISSIONS_URL.format(name=older.pop(0))))


def filing_documents(client: SecClient, filing: Filing) -> tuple[str, list[FilingDoc]]:
    """Read the filing index page; return its URL and the 10-K + EX-13 HTML documents."""
    index_url = ARCHIVES_URL.format(
        cik=filing.cik, acc=filing.folder, file=f"{filing.accession}-index.htm"
    )
    tree = lxml_html.fromstring(client.get(index_url).content)
    docs: list[FilingDoc] = []
    for row in tree.xpath('//table[contains(@class,"tableFile")][1]//tr[td]'):
        cells = row.xpath("./td")
        if len(cells) < 4:
            continue
        doc_type = cells[3].text_content().strip()
        href = (cells[2].xpath(".//a/@href") or [""])[0]
        name = href.split("/")[-1]
        if not name.lower().endswith((".htm", ".html")):
            continue
        if doc_type == filing.form or WANTED_EXHIBITS.match(doc_type):
            url = ARCHIVES_URL.format(cik=filing.cik, acc=filing.folder, file=name)
            docs.append(FilingDoc(type=doc_type, filename=name, url=url))
    if not any(d.type == filing.form for d in docs):  # fall back to submissions metadata
        name = filing.primary_document
        url = ARCHIVES_URL.format(cik=filing.cik, acc=filing.folder, file=name)
        docs.insert(0, FilingDoc(type=filing.form, filename=name, url=url))
    return index_url, docs


def download_company(
    client: SecClient, company: Company, out_dir: Path, dry_run: bool = False
) -> list[FilingMeta]:
    filings = list_10k_filings(client, company.cik, set(company.fiscal_years))
    missing = set(company.fiscal_years) - filings.keys()
    if missing:
        log.error("%s: no 10-K found for fiscal year(s) %s", company.ticker, sorted(missing))
    metas = []
    for fy in sorted(filings):
        f = filings[fy]
        dest = out_dir / company.ticker / f"FY{fy}"
        meta_path = dest / "meta.json"
        if meta_path.exists():
            meta = load_meta(meta_path)
            if all((dest / d.filename).exists() for d in meta.documents):
                log.info("%s FY%s already downloaded (%s)", company.ticker, fy, f.accession)
                metas.append(meta)
                continue
        index_url, docs = filing_documents(client, f)
        meta = FilingMeta(
            company=company.name,
            ticker=company.ticker,
            cik=company.cik,
            fiscal_year=fy,
            form=f.form,
            accession=f.accession,
            filing_date=f.filing_date,
            report_date=f.report_date,
            index_url=index_url,
            documents=docs,
        )
        log.info(
            "%s FY%s: %s filed %s, period %s, docs %s",
            company.ticker, fy, f.accession, f.filing_date, f.report_date,
            [d.type for d in docs],
        )
        if dry_run:
            metas.append(meta)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for d in docs:
            body = client.get(d.url).content
            (dest / d.filename).write_bytes(body)
            d.sha256 = hashlib.sha256(body).hexdigest()
            d.bytes = len(body)
        meta.downloaded_at = datetime.now(UTC).isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(asdict(meta), indent=2))
        metas.append(meta)
    return metas


def load_meta(path: Path) -> FilingMeta:
    raw = json.loads(path.read_text())
    raw["documents"] = [FilingDoc(**d) for d in raw["documents"]]
    return FilingMeta(**raw)


def iter_downloaded(raw_dir: Path) -> list[tuple[Path, FilingMeta]]:
    return [(p.parent, load_meta(p)) for p in sorted(raw_dir.glob("*/FY*/meta.json"))]
