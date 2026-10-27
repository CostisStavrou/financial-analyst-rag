# Financial Analyst RAG

*Started 27 Oct 2026. Target: ready to show in the Google Forward Deployed Engineering Intern (BS/MS, 2027, London) application, which closes 13 Nov 2026.*

A multi-agent RAG system that answers questions about US companies' annual reports (10-K filings), with exact numbers, page citations and a cross-check against the SEC's structured financial data. Built on Google Cloud with Gemini.

## Quick start (local, nothing paid on Google Cloud)

Needs Python 3.11+ and Docker.

```bash
git clone <this repo> && cd "Financial Analyst RAG"
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # then fill in SEC_USER_AGENT and GEMINI_API_KEY
docker compose up -d    # Postgres 17 + pgvector on localhost:5433
```

Then the pipeline, one command per step (run from the repo root):

```bash
.venv/bin/python -m finrag download   # 30 10-Ks from SEC EDGAR -> data/raw/edgar (~140 MB)
.venv/bin/python -m finrag parse      # sections, pages, chunks -> data/parsed/*.jsonl
.venv/bin/python -m finrag db-init    # schema (vector(768), HNSW + full-text indexes)
.venv/bin/python -m finrag index      # load chunks, embed with gemini-embedding-001
.venv/bin/python -m finrag ask "What is Amcor's year end FY2020 net AR?" --ticker AMCR --year 2020
```

Example (real run, 27 Oct; FinanceBench question 03882, answer $1,616m, evidence page 50):

```
Q: What is Amcor's year end FY2020 net AR (in USD millions)? ...
   5 hits | embed 701 ms | search 16 ms
[1] score 0.782 | Amcor 10-K FY2020, Item 8. Financial Statements and Supplementary Data, p. 50 (printed page 50) | table
    ... | Trade receivables, net | 1,615.9 | 1,864.3 | ...
    https://www.sec.gov/Archives/edgar/data/1748790/000174879020000025/amcr-20200630.htm
```

- `ask` prints the top-k chunks with a citation each: company, 10-K fiscal year, section, page (and the page number printed in the filing), plus the SEC URL. Filters: `--ticker`, `--year`, `--item 7`, `-k 10`, `--full`.
- `index --limit 50` embeds only 50 chunks (cheap smoke test). Embeddings are cached in `data/cache/embeddings.sqlite`, so re-indexing never pays twice.
- **Gemini quotas:** a paid key (Tier 1) embedded the whole corpus in about 4 minutes. On the free tier each chunk counts as one request: on 27 Oct it allowed about 100 chunks a minute and about 1,000 a day (`EmbedContentRequestsPerDay...FreeTier`). For a free key, `scripts/embed_daily.sh` runs once a day after the reset, resumes where it stopped, does the FinanceBench filings first and paces itself (`EMBED_RPM` / `EMBED_TPM`).
- No Gemini key? `EMBED_PROVIDER=local` with `pip install -e ".[local]"` uses BAAI/bge-base-en-v1.5 on CPU (768-dim, slow: ~1 chunk/s). Use a separate database for it, e.g. `DATABASE_URL=.../finrag_local`; the index refuses to mix models.
- Other commands: `stats` (chunks and embeddings per filing), `financebench` (questions covered), `page-check` (our pages vs FinanceBench evidence pages), `download --dry-run`, `db-init --reset`.
- Tests: `.venv/bin/python -m pytest`.

**macOS note:** if the repo sits in an iCloud-synced folder (e.g. `~/Documents`), macOS can mark files in `.venv` as hidden, and Python 3.13 then skips the editable-install `.pth` file, so the `finrag` console script fails with `No module named 'finrag'`. `python -m finrag` from the repo root always works.

## Status

| Days | Status |
|---|---|
| 1–4 (27–30 Oct) | **Done 27 Oct**: setup, EDGAR downloader, parser/chunker with pages and sections, all 12,103 chunks embedded with `gemini-embedding-001` in local pgvector, `ask` with citations working end to end. |
| 5–8 | Next: hybrid search (Postgres FTS + vector, RRF), reranking, FinanceBench baseline, table extraction. |

### Day 1–4 numbers (all from real runs on 27 Oct 2026)

- **Download:** 30 of 30 filings found and downloaded, 74 SEC requests, 140 MB. CVS FY2018 also has Exhibit 13 (annual report with MD&A and financials), which is fetched and parsed too.
- **Parse:** 3,871 pages → **12,103 chunks** (3,290 of them tables), ~3.57M tokens (estimated at 4 characters per token), in ~12 s.
- **Sections:** all 30 filings get their 10-K items detected (18–22 items each). Combined headings ("Items 10, 11, 12 and 13" at American Express) are filed under the first item; an out-of-order heading (PepsiCo FY2020 puts Item 9 after 9B) stays in the item before it.
- **Index:** all **12,103 chunks embedded** with `gemini-embedding-001` (768 dims) in local Postgres + pgvector. The paid-tier run did 11,403 chunks in **250 s** with 214 API requests and no rate limiting. The tokens embedded, counted beforehand with Gemini's free `count_tokens`, were 3,559,565, which is about **$0.53** at $0.15 per 1M. The 4-characters-per-token estimate gave 3,662,068 for the same chunks, 3% high.
- **Test queries** (FinanceBench questions; hits filtered to the right company and year unless stated):

  | Question | Evidence page (FinanceBench) | Where it came back | Figure in the chunk |
  |---|---|---|---|
  | Amcor FY2020 net receivables | 50 | hit 1, p. 50 | Trade receivables, net 1,615.9 (answer $1,616m) |
  | 3M FY2018 capital expenditure | 60 | hit 1, p. 60 | Purchases of PP&E (1,577) (answer $1,577m) |
  | AmEx effective tax rate FY2021→FY2022 | 44 | hit 3, p. 44 | 21.6% vs 24.6% (answer: down from 24.6% to 21.6%) |
  | Boeing FY2022 legal battles (**no filters**) | 113 | hits 2–3, p. 110–111 (2-page offset vs their PDF); all 5 hits Boeing | – |
  | General Mills FY2022 retention ratio | 45 and 49 | p. 49 (cash flow) hit 1; **p. 45 (income statement) not in the top 5** | – |

  Fresh query embedding took 347–701 ms and the vector search 12–39 ms (single runs). Repeated questions come from the cache in ~2 ms. The General Mills miss, a question that needs two statements, is the kind hybrid search and query decomposition should fix.
- **Offline model (bge-base on CPU):** works, but ~1 chunk/s on this 8 GB Mac, too slow for the full corpus.
- **Page check against FinanceBench** (`python -m finrag page-check`): for the 57 in-scope questions, 72 of 73 evidence passages were found in our chunks; **50 on the same page** as FinanceBench's PDF, **57 within ±1 page**. Most misses are offsets from cover pages in the company PDFs, plus CVS FY2018 where FinanceBench's PDF appends Exhibit 13 and we number it separately.

## How ingestion works

- **Download** ([`finrag/ingest/edgar.py`](finrag/ingest/edgar.py)): `data.sec.gov/submissions` for the filing list, the filing's `-index.htm` for its documents, then the 10-K HTML (plus any `EX-13`). Original 10-Ks only, no amendments. Fiscal year = the year the period ends (Best Buy FY2023 ended 28 Jan 2023), checked against FinanceBench's own accession numbers for 3M and AMD.
- **Parse** ([`finrag/ingest/parse_html.py`](finrag/ingest/parse_html.py)): HTML → ordered paragraphs and tables. Pages come from the filing's CSS page breaks; the printed page number in the footer is kept as `page_label`. Hidden inline-XBRL headers are dropped. 10-K items are found from "Item 7." style headings, ignoring the table of contents. Filings converted from PDF (General Mills FY2022: every line an absolutely positioned `<div>`, no `<table>` tags) are rebuilt from x/y coordinates into lines, paragraphs and table rows.
- **Chunk** ([`finrag/ingest/chunk.py`](finrag/ingest/chunk.py)): ~500-token text chunks within one section, a short paragraph of overlap; every table is its own chunk with its caption (long tables split by rows, header repeated). Each chunk keeps company, ticker, fiscal year, accession, document, item, section, page range, printed page and SEC URL. The embedded text starts with a one-line header (company, year, section, page).
- **Index** ([`finrag/index/`](finrag/index)): Postgres `chunks` table with `vector(768)` (HNSW, cosine) and a generated `tsvector` column (GIN) ready for hybrid search.

## Repo layout

```
config/companies.toml   corpus: companies, CIKs, fiscal years
finrag/ingest/          EDGAR download, HTML parsing, chunking
finrag/index/           embeddings (Gemini / local), Postgres schema and loading
finrag/retrieve/        vector search with metadata filters
finrag/agents/          planner, retriever, calculator, verifier (days 9-12)
finrag/eval/            FinanceBench loading, coverage, page check
finrag/cli.py           python -m finrag <command>
scripts/embed_daily.sh  daily free-tier Gemini embedding run
tests/                  parser, chunker and embedder tests
docker-compose.yml      local Postgres + pgvector
```

## Why

- Annual reports are 100–300 pages each. Across dozens of companies and years, that's far beyond any chat context: retrieval is genuinely needed.
- It shows what the Google posting asks for: RAG pipelines, vector databases, multi-agent frameworks, LLM cost and latency metrics, shipping into a cloud environment.
- It is measurable: **FinanceBench** gives real questions with correct answers.

## Data

- **SEC EDGAR**: 10-K filings via the official free API. SEC rules: identify yourself in the `User-Agent` header (name + contact email) and stay under their request-rate limit. The downloader waits ≥ 0.5 s between requests (≤ 2/s; SEC's limit is 10/s).
- **SEC XBRL "company facts"**: structured financial figures, used to cross-check numbers (days 9–12).
- **FinanceBench**: [`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench) on Hugging Face, the open-source sample of 150 questions (112 on 10-Ks). **Licence: CC BY-NC 4.0** (checked 27 Oct 2026): fine for this non-commercial portfolio project with attribution (Islam et al., 2023, [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)). It is downloaded on demand into `data/` and not redistributed in this repo.

### Corpus: 10 companies × 3 fiscal years

Chosen to overlap with FinanceBench: for each company, the 3 fiscal years with the most 10-K questions, then the top 10 companies (CVS Health instead of Corning, tied, to add a Health Care name). Gaps filled with the nearest earlier year. Defined in [`config/companies.toml`](config/companies.toml).

| Company | Ticker | Fiscal years | FinanceBench 10-K questions covered |
|---|---|---|---|
| AMD | AMD | 2015, 2021, 2022 | 8 |
| Boeing | BA | 2018, 2021, 2022 | 8 |
| American Express | AXP | 2020, 2021, 2022 | 7 |
| PepsiCo | PEP | 2020, 2021, 2022 | 6 |
| 3M | MMM | 2018, 2021, 2022 | 5 |
| Amcor | AMCR | 2020, 2022, 2023 | 5 |
| Best Buy | BBY | 2017, 2019, 2023 | 5 |
| Verizon | VZ | 2020, 2021, 2022 | 5 |
| CVS Health | CVS | 2018, 2021, 2022 | 4 |
| General Mills | GIS | 2019, 2020, 2022 | 4 |

**57 of the 112** FinanceBench 10-K questions fall inside this corpus (`python -m finrag financebench`). The evaluation runs on those 57.

## Architecture (planned)

1. **Ingest**: download 10-Ks, parse sections, extract **tables as structured data** (Document AI or Gemini), keep page numbers.
2. **Index**: chunks + embeddings (`gemini-embedding-001`) in **pgvector** (AlloyDB or Cloud SQL), plus Postgres full-text search, plus metadata (company, year, section, page).
3. **Retrieve**: hybrid search (keyword + vector, Reciprocal Rank Fusion), then reranking.
4. **Agents** (Google Agent Development Kit): planner → retriever → calculator → verifier. Model routing: cheaper Gemini model for simple steps, stronger one for reasoning.
5. **Verify**: compare numbers found in the text with SEC XBRL figures; flag disagreements; say "I don't know" when the answer isn't there.
6. **Serve**: Cloud Run web app; answers cite company, year and page.

## Evaluation

Run FinanceBench on each version and report accuracy, cost per question and latency:

| Version | Accuracy | Cost / question | Latency |
|---|---|---|---|
| Vector only | – | – | – |
| Hybrid | – | – | – |
| Hybrid + rerank | – | – | – |
| Hybrid + rerank + tables | – | – | – |
| Agentic (+ XBRL check) | – | – | – |

Plus error analysis: wrong page retrieved, table misread, maths error, refusal.

## Priorities

1. Evaluation and comparison table (must)
2. Proper table extraction (must)
3. Deployed on Google Cloud, customer-ready README, one-command deploy (must)
4. Multi-agent design with cost/latency tracking and model routing (must)
5. SEC XBRL cross-check (must)
6. Live demo link + 2–3 minute video (must)

Nice to have: clickable page citations, measured "I don't know", automatic updates for new filings, Cloud Trace.
Cut for now: fancy UI, more than ~10 companies, fine-tuning, extra data sources.

## Schedule

| Days | Dates | Build |
|---|---|---|
| 1–4 | 27–30 Oct | Project setup; download 10-Ks; parse, chunk (with page numbers), embed, store in pgvector |
| 5–8 | 31 Oct – 3 Nov | Hybrid search + reranking; FinanceBench baseline; table extraction |
| 9–12 | 4–7 Nov | Agents (planner, retriever, calculator, verifier); XBRL cross-check; cost/latency tracking |
| 13–15 | 8–10 Nov | Deploy on Cloud Run; README with architecture and results; demo video |
| 16–17 | 11–12 Nov | Final numbers; apply (deadline 13 Nov) |

## Rules

- API keys and credentials only in `.env` (never committed); `.gitignore` from day one.
- No paid Google Cloud resources created without Kostis's go-ahead.
- Every reported number comes from a real evaluation run.
