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
.venv/bin/python -m finrag eval-retrieval   # FinanceBench: is the evidence in the top 20?
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
- `index --limit 50` embeds only 50 chunks (cheap smoke test). Embeddings are cached in `data/cache/embeddings.sqlite`, so re-indexing never pays twice. After writing new vectors, `index` rebuilds the HNSW index (4 s for the full corpus); `index --reindex` forces a rebuild ([why](#hnsw-vs-exact-search)).
- **Gemini quotas:** a paid key (Tier 1) embedded the whole corpus in about 4 minutes. On the free tier each chunk counts as one request: on 27 Oct it allowed about 100 chunks a minute and about 1,000 a day (`EmbedContentRequestsPerDay...FreeTier`). For a free key, `scripts/embed_daily.sh` runs once a day after the reset, resumes where it stopped, does the FinanceBench filings first and paces itself (`EMBED_RPM` / `EMBED_TPM`).
- No Gemini key? `EMBED_PROVIDER=local` with `pip install -e ".[local]"` uses BAAI/bge-base-en-v1.5 on CPU (768-dim, slow: ~1 chunk/s). Use a separate database for it, e.g. `DATABASE_URL=.../finrag_local`; the index refuses to mix models.
- Other commands: `stats` (chunks and embeddings per filing), `financebench` (questions covered), `page-check` (our pages vs FinanceBench evidence pages), `download --dry-run`, `db-init --reset`.
- Tests: `.venv/bin/python -m pytest`.

**macOS note:** if the repo sits in an iCloud-synced folder (e.g. `~/Documents`), macOS can mark files in `.venv` as hidden, and Python 3.13 then skips the editable-install `.pth` file, so the `finrag` console script fails with `No module named 'finrag'`. `python -m finrag` from the repo root always works.

## Status

| Days | Status |
|---|---|
| 1–4 (27–30 Oct) | **Done 27 Oct**: setup, EDGAR downloader, parser/chunker with pages and sections, all 12,103 chunks embedded with `gemini-embedding-001` in local pgvector, `ask` with citations working end to end. |
| 5–8 | **Started 27 Oct**: vector-only retrieval baseline on FinanceBench ([results](#retrieval-no-llm-is-the-evidence-in-the-top-k)). Next: hybrid search (Postgres FTS + vector, RRF), reranking, table extraction. |

### Day 1–4 numbers (all from real runs on 27 Oct 2026)

- **Download:** 30 of 30 filings found and downloaded, 74 SEC requests, 140 MB. CVS FY2018 also has Exhibit 13 (annual report with MD&A and financials), which is fetched and parsed too.
- **Parse:** 3,871 pages → **12,103 chunks** (3,290 of them tables), ~3.57M tokens (estimated at 4 characters per token), in ~12 s.
- **Sections:** all 30 filings get their 10-K items detected (18–22 items each). Combined headings ("Items 10, 11, 12 and 13" at American Express) are filed under the first item; an out-of-order heading (PepsiCo FY2020 puts Item 9 after 9B) stays in the item before it.
- **Index:** all **12,103 chunks embedded** with `gemini-embedding-001` (768 dims) in local Postgres + pgvector. The paid-tier run did 11,403 chunks in **250 s** with 214 API requests and no rate limiting. The 11,453 chunks embedded on the paid tier (a 50-chunk check, then this run) were 3,559,565 tokens, counted beforehand with Gemini's free `count_tokens`: about **$0.53** at $0.15 per 1M (the model's launch price; Google's pricing page no longer lists `gemini-embedding-001` on 27 Oct 2026). The 4-characters-per-token estimate gave 3,662,068 for the same chunks, 3% high.
- **Test queries** (FinanceBench questions; hits filtered to the right company and year unless stated):

  | Question | Evidence page (FinanceBench) | Where it came back | Figure in the chunk |
  |---|---|---|---|
  | Amcor FY2020 net receivables | 50 | hit 1, p. 50 | Trade receivables, net 1,615.9 (answer $1,616m) |
  | 3M FY2018 capital expenditure | 60 | hit 1, p. 60 | Purchases of PP&E (1,577) (answer $1,577m) |
  | AmEx effective tax rate FY2021→FY2022 | 44 | hit 3, p. 44 | 21.6% vs 24.6% (answer: down from 24.6% to 21.6%) |
  | Boeing FY2022 legal battles (**no filters**) | 113 | hits 2–3, p. 110–111 (2-page offset vs their PDF); all 5 hits Boeing | – |
  | General Mills FY2022 retention ratio | 45 and 49 | p. 49 (cash flow) hit 1; **p. 45 (income statement) not in the top 5** | – |

  Fresh query embedding took 259–701 ms and the vector search 12–39 ms (single runs). Repeated questions come from the cache in ~2 ms. The General Mills miss, a question that needs two statements, is the kind hybrid search and query decomposition should fix.
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
finrag/eval/            FinanceBench loading, coverage, page check, retrieval scoring
finrag/cli.py           python -m finrag <command>
results/                evaluation results: per-question ranks and chunk ids, no FinanceBench text
scripts/embed_daily.sh  daily free-tier Gemini embedding run
tests/                  parser, chunker, embedder and retrieval-metric tests
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

Run FinanceBench on each version and report retrieval, answer accuracy, cost per question and latency (57 in-scope questions):

| Version | Evidence in top 5 (no filter) | Answer accuracy | Cost / question | Latency (median) |
|---|---|---|---|---|
| Vector only | **86%** (49/57) | – | $0.000006 (query embedding only) | 279 ms embed + 3 ms search |
| Hybrid | – | – | – | – |
| Hybrid + rerank | – | – | – | – |
| Hybrid + rerank + tables | – | – | – | – |
| Agentic (+ XBRL check) | – | – | – | – |

Plus error analysis: wrong page retrieved, table misread, maths error, refusal.

### Retrieval (no LLM): is the evidence in the top k?

`python -m finrag eval-retrieval` (`--exact` for brute-force search). Real runs on 27 Oct 2026: 57 questions, the FinanceBench question verbatim as the query, top 20 chunks. Code: [`finrag/eval/retrieval.py`](finrag/eval/retrieval.py).

- **Gold labels.** Each question has 1–3 evidence passages from FinanceBench (73 in all). Each passage is matched to our chunks in the question's filing using 60-character windows of normalised text (lowercase, letters and digits only). Gold chunks are:
  - the chunk sharing the most windows;
  - any chunk repeating at least half the passage (the same sentence in MD&A and the notes);
  - other chunks on those pages sharing at least 10% of the windows (rounded up, never more than 3).

  A table header shared with a table on another page doesn't count. All 73 passages match. The page check's one miss, 3M FY2022, is found now: our chunk lacks the title that opens the passage, and the denser windows (every 30 characters, the last ending at the end) fall inside the rest. Passages under 40 normalised characters would count as not found; the shortest here is 58.

  Limit: gold only sees word-for-word copies. A reworded passage, or the same figures under a different table header, is not gold even when it answers the question ([examples](#where-vector-only-retrieval-misses)).
- **Metrics.**
  - **Page hit@k** (main): a top-k chunk is on a gold page of the right filing.
  - **Chunk hit@k**: a gold chunk itself is in the top k.
  - **MRR**: 1 / rank of the first page hit.
  - **All passages@k**: for the 13 questions with several passages, every passage is found.
- **Settings.**
  - No filter: all 30 filings.
  - Company filter.
  - Company + fiscal year: the right filing, as if an oracle picked it.

| Setting | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | MRR | Chunk hit@5 | All passages @5 / @20 |
|---|---|---|---|---|---|---|---|---|
| No filter | 53% (30) | 77% (44) | **86% (49)** | 95% (54) | 96% (55) | 0.67 | 84% (48) | 4 / 9 of 13 |
| Company | same as no filter | | | | | | | |
| Company + year | 54% (31) | 82% (47) | **95% (54)** | 96% (55) | 98% (56) | 0.70 | 91% (52) | 4 / 9 of 13 |

- **The company filter changes nothing.** The top 20 is identical with and without it for all 57 questions. Unfiltered search never left the right company: all 285 top-5 chunks came from it, probably because the question names the company and so does every chunk's embedded header (not tested). It picked the right fiscal year for 220 of the 285 (top 1: 54 of 57).
- **By question type** (no filter, hit@5):
  - metrics-generated: 15/15
  - novel-generated: 8/10
  - domain-relevant: 26/32

  Weakest company: Boeing, 4/8 (5/8 with the right filing).
- **Several statements are the clearest gap.** 9 of the 13 questions with several passages need two or three statements, e.g. income statement + balance sheet.
  - Every statement is in the top 5 for only 2 of the 9 (AMD FY2015, PepsiCo FY2022 EBITDA margin).
  - With the right filing, the missing statement comes back at hits 6–10 for 5 of them. For 3M and CVS "capital intensive?" the income statement is not in the top 20.
  - The other 4 are text questions; 2 have every passage in the top 5, hence 4 of 13 in the table.
- **Latency and cost** (median / p95, single runs on this 8 GB Mac; timings vary from run to run):
  - Query embedding: 279 / 425 ms (52 Gemini calls in the first run; later runs read the cache).
  - Search, no filter: 3 / 8 ms with HNSW, 64 / 70 ms exact.
  - Search, company + year: 4 / 5 ms (Postgres sorts the filing's rows exactly, [see below](#hnsw-vs-exact-search)).
  - Cost: the 57 questions are 2,097 tokens by Gemini's `count_tokens` (27 Oct), about $0.000006 per question at $0.15 per 1M, `gemini-embedding-001`'s launch price. Google's pricing page no longer lists this model; at `gemini-embedding-2`'s $0.20 it would be $0.000007.
- **Results** (per-question ranks and chunk ids, no FinanceBench text):
  - [`results/retrieval_vector.json`](results/retrieval_vector.json)
  - [`results/retrieval_vector_exact.json`](results/retrieval_vector_exact.json)
  - [`results/retrieval_vector_before_reindex.json`](results/retrieval_vector_before_reindex.json)

#### HNSW vs exact search

- **First run: HNSW lost one question.** With no filter, the HNSW index missed 21 of the 1,140 exact top-20 results (57 questions × 20). 20 of them were from one question: AMD FY2022, operating margin. Exact search ranks the evidence 1st. HNSW returned none of AMD's top 20 and got stuck among CVS, General Mills and Verizon chunks. Effect: hit@5 84% instead of 86%.
- **Likely cause: how the index was built.** It had been grown row by row during embedding, and every `index` run rewrote all rows: 73,933 updates for 12,103 chunks (counted before a later database restart reset the statistics). A fresh build of the same vectors in a scratch table missed 1 of 1,140. Row-by-row growth and rewrites were not tested separately.
- **Fix.**
  - `index` no longer rewrites unchanged rows (checked: a re-run left the update count unchanged).
  - It rebuilds the HNSW index after writing vectors.
- **After the rebuild**, HNSW returns the exact top 20, in the same order, for all 57 no-filter searches (1,140 of 1,140). Searches also got faster: median 22 ms before, 3 ms after (p95 104 → 8 ms).
- **With filters, Postgres mostly skips HNSW.** `eval-retrieval` records each search's plan (`EXPLAIN`):
  - company filter: HNSW for 21 of 57 searches (AXP, CVS, MMM, VZ);
  - company + year: none. Postgres reads the filing's rows and sorts them, which is exact.

  So the HNSW-vs-exact check only covers the 57 no-filter searches and those 21. All of them match.

#### Where vector-only retrieval misses

**The gold labels miss some real hits.** Checked by hand, 10 questions have the answer at hit 1 (no filter) but score lower:

- Amcor gross margin and Boeing effective tax rate: in the table below (not in the top 5).
- Boeing FY2022 segment revenue: the MD&A revenue table, with the same figures as the gold Notes table (scored at hit 4).
- AMD FY2022 products: the same product lines in other words (hit 5).
- American Express tax-rate change: the tax note, 24.6% → 21.6% (hit 5).
- 3M FY2022 operating-margin drivers: the MD&A margin bridge (hit 3).
- Hit 2, with the facts in another part of the filing:
  - AMD FY2022 cash-flow mix: MD&A summary table;
  - Best Buy acquisitions: MD&A text;
  - Best Buy FY2015–17 net margin: selected financial data;
  - Boeing FY2018 net PP&E: PP&E note, $12,645.

Counting these 10 would give hit@1 40/57 (70%) and hit@5 51/57 (89%). The tables keep the automatic labels, so every version is scored the same way.

The 8 questions with no evidence page in the no-filter top 5:

| Why | Questions | Note |
|---|---|---|
| Answer retrieved, but not from FinanceBench's page | Amcor FY2023 gross margin: MD&A table, 18.5% vs 19.4%, hits 1–2. Boeing FY2022 effective tax rate: tax note, (0.6)% vs 14.7%, hit 1. | The gold is one valid source; an LLM could answer from these. |
| Wrong year; the question gives none | Best Buy gross margins "historically": FY2017 selected data at hit 1. Boeing cyclicality: the same risk-factor sentence from FY2018 and FY2021 at hits 3 and 5. | Hit 2 for both with the year filter. |
| Needs several statements | 3M "capital intensive?": income statement, balance sheet, cash flow. Verizon "capital intensive?": balance sheet + MD&A revenue table. | 3M: a segment table first; cash flow and balance sheet at hits 6 and 7, income statement not in the top 20. Verizon: the cash-flow statement first; balance sheet at hit 11; the revenue table not in the top 20 (hit 6 only shares its page). |
| Wrong part of the right filing | Boeing gross margin: segment operating-margin tables came first, evidence at hit 10. Boeing primary customers: the key sentences sit inside 350–500-token risk-factor chunks; not in the top 20. | |

**Next:**

- Query decomposition (one sub-query per statement) for ratio questions: 3M's and CVS's income statements and Verizon's revenue table are not in the top 20.
- Reranking the top 20, or smaller chunks, for sentences buried in long chunks (Boeing customers).
- Hybrid search (keywords + vectors), to be measured. It can only help where the question shares words with the filing: the gross-margin evidence says "Gross profit", and vector search already puts both tax-rate notes at hit 1.
- Extracting the company and year from the question: the right filing lifts hit@5 from 86% to 95%.

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
