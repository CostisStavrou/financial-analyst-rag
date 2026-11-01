# How it is built

Details behind the [README](../README.md): how filings become searchable chunks, the corpus, and practical notes for running it.

## Ingestion

- **Download** ([`finrag/ingest/edgar.py`](../finrag/ingest/edgar.py)): `data.sec.gov/submissions` for the filing list, the filing's `-index.htm` for its documents, then the 10-K HTML (plus any `EX-13`). Original 10-Ks only, no amendments. Fiscal year = the year the period ends (Best Buy FY2023 ended 28 Jan 2023), checked against FinanceBench's own accession numbers for 3M and AMD.
- **Parse** ([`finrag/ingest/parse_html.py`](../finrag/ingest/parse_html.py)): HTML → ordered paragraphs and tables. Pages come from the filing's CSS page breaks; the printed page number in the footer is kept as `page_label`. Hidden inline-XBRL headers are dropped. 10-K items are found from "Item 7." style headings, ignoring the table of contents. Filings converted from PDF (General Mills FY2022: every line an absolutely positioned `<div>`, no `<table>` tags) are rebuilt from x/y coordinates into lines, paragraphs and table rows.
- **Chunk** ([`finrag/ingest/chunk.py`](../finrag/ingest/chunk.py)): ~500-token text chunks within one section, a short paragraph of overlap; every table is its own chunk with its caption (long tables split by rows, header repeated). Each chunk keeps company, ticker, fiscal year, accession, document, item, section, page range, printed page and SEC URL. The embedded text starts with a one-line header (company, year, section, page).
- **Index** ([`finrag/index/`](../finrag/index)): Postgres `chunks` table with `vector(768)` (HNSW, cosine), plus materialised views for BM25 keyword search (`keyword_docs`, `keyword_postings`), rebuilt by `finrag index`.

## Corpus: 10 companies × 3 fiscal years

Chosen to overlap with FinanceBench: for each company, the 3 fiscal years with the most 10-K questions, then the top 10 companies (CVS Health instead of Corning, tied, to add a Health Care name). Gaps filled with the nearest earlier year. Defined in [`config/companies.toml`](../config/companies.toml).

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

## Ingestion numbers (real runs on 27 Oct 2026)

- **Download:** 30 of 30 filings found and downloaded, 74 SEC requests, 140 MB. CVS FY2018 also has Exhibit 13 (annual report with MD&A and financials), which is fetched and parsed too.
- **Parse:** 3,871 pages → **12,103 chunks** (3,290 of them tables), ~3.57M tokens (estimated at 4 characters per token), in ~12 s.
- **Sections:** all 30 filings get their 10-K items detected (18–22 items each). Combined headings ("Items 10, 11, 12 and 13" at American Express) are filed under the first item; an out-of-order heading (PepsiCo FY2020 puts Item 9 after 9B) stays in the item before it.
- **Index:** all **12,103 chunks embedded** with `gemini-embedding-001` (768 dims) in local Postgres + pgvector. The paid-tier run did 11,403 chunks in **250 s** with 214 API requests and no rate limiting. The 11,453 chunks embedded on the paid tier (a 50-chunk check, then this run) were 3,559,565 tokens, counted beforehand with Gemini's free `count_tokens`: about **$0.53** at $0.15 per 1M (the model's launch price; Google's pricing page no longer lists `gemini-embedding-001` on 27 Oct 2026). The 4-characters-per-token estimate gave 3,662,068 for the same chunks, 3% high.
- **Test queries** (FinanceBench questions, paraphrased; hits filtered to the right company and year unless stated):

  | Question | Evidence page (FinanceBench) | Where it came back | Figure in the chunk |
  |---|---|---|---|
  | Amcor FY2020 net receivables | 50 | hit 1, p. 50 | Trade receivables, net 1,615.9 (answer $1,616m) |
  | 3M FY2018 capital expenditure | 60 | hit 1, p. 60 | Purchases of PP&E (1,577) (answer $1,577m) |
  | AmEx effective tax rate FY2021→FY2022 | 44 | hit 3, p. 44 | 21.6% vs 24.6% (answer: down from 24.6% to 21.6%) |
  | Boeing FY2022 legal battles (**no filters**) | 113 | hits 2–3, p. 110–111 (2-page offset vs their PDF); all 5 hits Boeing | – |
  | General Mills FY2022 retention ratio | 45 and 49 | p. 49 (cash flow) hit 1; **p. 45 (income statement) not in the top 5** | – |

  Fresh query embedding took 259–701 ms and the vector search 12–39 ms (single runs). Repeated questions come from the cache in ~2 ms. The General Mills miss, a question that needs two statements, is the kind query decomposition should fix (hybrid search, [measured](EVALUATION.md#hybrid-search-keyword--vector), doesn't help).
- **Offline model (bge-base on CPU):** works, but ~1 chunk/s on this 8 GB Mac, too slow for the full corpus.
- **Page check against FinanceBench** (`python -m finrag page-check`): for the 57 in-scope questions, 72 of 73 evidence passages were found in our chunks; **50 on the same page** as FinanceBench's PDF, **57 within ±1 page**. Most misses are offsets from cover pages in the company PDFs, plus CVS FY2018 where FinanceBench's PDF appends Exhibit 13 and we number it separately.

## Running it: commands and notes

- `ask` prints the top-k chunks with a citation each: company, 10-K fiscal year, section, page (and the page number printed in the filing), plus the SEC URL. Filters: `--ticker`, `--year`, `--item 7`, `-k 10`, `--full`; `--method keyword|hybrid` for the other searches.
- `index --limit 50` embeds only 50 chunks (cheap smoke test). Embeddings are cached in `data/cache/embeddings.sqlite`, so re-indexing never pays twice. After writing new vectors, `index` rebuilds the HNSW index (4 s for the full corpus); `index --reindex` forces a rebuild ([why](EVALUATION.md#hnsw-vs-exact-search)).
- **Gemini quotas:** a paid key (Tier 1) embedded 11,403 chunks in 250 s; the rest were embedded earlier on the free tier. On the free tier each chunk counts as one request: on 27 Oct it allowed about 100 chunks a minute and about 1,000 a day (`EmbedContentRequestsPerDay...FreeTier`). For a free key, `scripts/embed_daily.sh` runs once a day after the reset, resumes where it stopped, does the FinanceBench filings first and paces itself (`EMBED_RPM` / `EMBED_TPM`).
- No Gemini key? `EMBED_PROVIDER=local` with `pip install -e ".[local]"` uses BAAI/bge-base-en-v1.5 on CPU (768-dim, slow: ~1 chunk/s). Use a separate database for it, e.g. `DATABASE_URL=.../finrag_local`; the index refuses to mix models.
- Other commands: `stats` (chunks and embeddings per filing), `financebench` (questions covered), `page-check` (our pages vs FinanceBench evidence pages), `download --dry-run`, `db-init --reset`.
- Tests: `.venv/bin/python -m pytest`.

**macOS note:** if the repo sits in an iCloud-synced folder (e.g. `~/Documents`), macOS can mark files in `.venv` as hidden, and Python 3.13 then skips the editable-install `.pth` file, so the `finrag` console script fails with `No module named 'finrag'`. `python -m finrag` from the repo root always works.

## Design notes

- **Vector search is the default.** Hybrid search is built and can be selected (`ask --method hybrid`), but it lowered hit@5 from 86% to 53% ([why](EVALUATION.md#results-hybrid-is-worse-than-vector-alone)).
- **BM25 in plain SQL**, with no Postgres extension beyond pgvector, so it would run unchanged on a managed Postgres such as Cloud SQL or AlloyDB.
- **Model routing in the agents:** the cheapest model (`gemini-3.1-flash-lite`) plans from the question alone; `gemini-3.8-flash` reads the sources.
- **Costs are tracked per call** from the API's own token counts, with an optional spending cap per evaluation run (`--max-usd`) and a reply cache so re-runs cost nothing.
