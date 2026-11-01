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
.venv/bin/python -m finrag eval-retrieval --method hybrid   # same, for BM25 + vector (RRF)
.venv/bin/python -m finrag answer "How many people did PepsiCo employ at the end of 2022?"   # Gemini answer with sources
.venv/bin/python -m finrag eval-answers --estimate   # token count and cost of the answer-accuracy run, no paid calls
.venv/bin/python -m finrag agent "By what percentage did PepsiCo's net revenue grow from fiscal 2021 to fiscal 2022?"   # multi-agent answer, cost per step
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
| 5–8 | **Started 27 Oct**: vector-only retrieval baseline on FinanceBench ([results](#retrieval-no-llm-is-the-evidence-in-the-top-k)); hybrid search (BM25 in Postgres + vector, RRF) built and measured: worse than vector alone ([results](#results-hybrid-is-worse-than-vector-alone)). **31 Oct:** answer accuracy with Gemini, 95% with no filter ([results](#results-95-correct-with-no-filter)). **1 Nov:** multi-agent pipeline on Google ADK, 91% at 8× the cost ([results](#results-no-better-than-one-call-at-8-the-cost)). Next: a harder test set, XBRL cross-check. |

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

  Fresh query embedding took 259–701 ms and the vector search 12–39 ms (single runs). Repeated questions come from the cache in ~2 ms. The General Mills miss, a question that needs two statements, is the kind query decomposition should fix (hybrid search, [measured later](#hybrid-search-keyword--vector), doesn't help).
- **Offline model (bge-base on CPU):** works, but ~1 chunk/s on this 8 GB Mac, too slow for the full corpus.
- **Page check against FinanceBench** (`python -m finrag page-check`): for the 57 in-scope questions, 72 of 73 evidence passages were found in our chunks; **50 on the same page** as FinanceBench's PDF, **57 within ±1 page**. Most misses are offsets from cover pages in the company PDFs, plus CVS FY2018 where FinanceBench's PDF appends Exhibit 13 and we number it separately.

## How ingestion works

- **Download** ([`finrag/ingest/edgar.py`](finrag/ingest/edgar.py)): `data.sec.gov/submissions` for the filing list, the filing's `-index.htm` for its documents, then the 10-K HTML (plus any `EX-13`). Original 10-Ks only, no amendments. Fiscal year = the year the period ends (Best Buy FY2023 ended 28 Jan 2023), checked against FinanceBench's own accession numbers for 3M and AMD.
- **Parse** ([`finrag/ingest/parse_html.py`](finrag/ingest/parse_html.py)): HTML → ordered paragraphs and tables. Pages come from the filing's CSS page breaks; the printed page number in the footer is kept as `page_label`. Hidden inline-XBRL headers are dropped. 10-K items are found from "Item 7." style headings, ignoring the table of contents. Filings converted from PDF (General Mills FY2022: every line an absolutely positioned `<div>`, no `<table>` tags) are rebuilt from x/y coordinates into lines, paragraphs and table rows.
- **Chunk** ([`finrag/ingest/chunk.py`](finrag/ingest/chunk.py)): ~500-token text chunks within one section, a short paragraph of overlap; every table is its own chunk with its caption (long tables split by rows, header repeated). Each chunk keeps company, ticker, fiscal year, accession, document, item, section, page range, printed page and SEC URL. The embedded text starts with a one-line header (company, year, section, page).
- **Index** ([`finrag/index/`](finrag/index)): Postgres `chunks` table with `vector(768)` (HNSW, cosine), plus materialised views for BM25 keyword search (`keyword_docs`, `keyword_postings`), rebuilt by `finrag index`.

## Repo layout

```
config/companies.toml   corpus: companies, CIKs, fiscal years
finrag/ingest/          EDGAR download, HTML parsing, chunking
finrag/index/           embeddings (Gemini / local), Postgres schema and loading
finrag/retrieve/        vector search with metadata filters, BM25 keyword search, hybrid (RRF)
finrag/agents/          ADK pipeline: planner, retriever, analyst with calculator tool, verifier
finrag/answer/          Gemini answers with citations (cached, with token and cost accounting)
finrag/eval/            FinanceBench loading, coverage, page check, retrieval and answer scoring
finrag/cli.py           python -m finrag <command>
results/                evaluation results: per-question ranks and chunk ids, no FinanceBench text
scripts/embed_daily.sh  daily free-tier Gemini embedding run
tests/                  parser, chunker, embedder, retrieval-metric, fusion and answer tests (no API calls); BM25 tests run against the local database when it is up
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
3. **Retrieve**: vector search with company/year filters, then reranking. Hybrid search (BM25 + vector, Reciprocal Rank Fusion) is built but lowered hit@5 from 86% to 53% on FinanceBench ([why](#results-hybrid-is-worse-than-vector-alone)), so it is off by default.
4. **Agents** (Google Agent Development Kit): planner → retriever → calculator → verifier. Model routing: cheaper Gemini model for simple steps, stronger one for reasoning.
5. **Verify**: compare numbers found in the text with SEC XBRL figures; flag disagreements; say "I don't know" when the answer isn't there.
6. **Serve**: Cloud Run web app; answers cite company, year and page.

## Evaluation

Run FinanceBench on each version and report retrieval, answer accuracy, cost per question and latency (57 in-scope questions):

| Version | Evidence in top 5 (no filter) | Answer accuracy | Cost / question | Latency (median) |
|---|---|---|---|---|
| Vector only | **86%** (49/57) | **95%** (54/57; 91% by both graders), top 10 chunks, `gemini-3.8-flash` ([results](#results-95-correct-with-no-filter)) | $0.0077 (answer; retrieval $0.000006) | 279 ms embed + 3 ms search + 7.5 s answer |
| Hybrid (BM25 + vector, RRF) | 53% (30/57) ([why lower](#results-hybrid-is-worse-than-vector-alone)) | – | $0.000006 (query embedding only) | 279 ms embed + 124 ms search |
| Vector + rerank | – | – | – | – |
| Vector + rerank + tables | – | – | – | – |
| Agents (ADK: planner, retriever, analyst + calculator, verifier) | – (evidence page in the sources for 50/57) | 91% (52/57) ([results](#results-no-better-than-one-call-at-8-the-cost)) | $0.064 | 32.7 s |
| Agents + XBRL check | – | – | – | – |

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
- Hybrid search (keywords + vectors): measured [below](#hybrid-search-keyword--vector). It lowers hit@5 from 86% to 53%.
- Extracting the company and year from the question: the right filing lifts hit@5 from 86% to 95%.

### Hybrid search: keyword + vector

`python -m finrag eval-retrieval --method hybrid` (`ask --method keyword|hybrid` for single questions). Code: [`finrag/retrieve/keyword.py`](finrag/retrieve/keyword.py), [`finrag/retrieve/hybrid.py`](finrag/retrieve/hybrid.py).

#### Decided before the first run

The 57 questions are the only evaluation set, so every choice below was fixed, after three independent design reviews, before any keyword or hybrid result was seen. Nothing is tuned on the 57. A change made after seeing results would be a new, labelled row.

- **Keyword leg: BM25 in plain SQL.** No extension beyond pgvector, so it runs unchanged on Cloud SQL and AlloyDB.
  - Lucene's formula and defaults: k1 1.2, b 0.75, idf = ln(1 + (N − df + 0.5) / (df + 0.5)). The classic idf turns negative for words in over half the chunks ("item", "statement").
  - It searches the same text the embedding model sees (`Chunk.embed_text`: company, filing, fiscal year, section and page, then the content).
  - Postgres's English text search (stemming, stop words); a chunk matches on any of the question's words. Postgres's own `plainto_tsquery` / `websearch_to_tsquery` need every word and match no chunk for a typical question.
  - N, document frequency and average length cover the whole corpus, also with filters, so a filtered ranking is the unfiltered one restricted to the filter. A test checks the SQL against a plain-Python BM25.
  - Stored as materialised views (`keyword_docs`, `keyword_postings`), rebuilt by `finrag index` in about 6 s. The `chunks` table and its HNSW index are not rewritten, so the vector baseline stays as measured.
- **Fusion: Reciprocal Rank Fusion**, k = 60 (Cormack et al., 2009), the top 50 of each leg, equal weights, top 20 kept.
  - Ties: equal scores within a leg share a rank; equal fused scores go to the better vector rank, then the better keyword rank, then a hash of the chunk id. Chunk ids start with ticker and year, so sorting by id would favour some companies and older filings.
- **What this predicts.** With k 60 and 50 per leg, any chunk in both lists outranks any chunk in only one. Where the legs disagree, hybrid can push the vector leg's first hit down, so hit@1 and MRR can fall while hit@20 rises.
- **Handicaps left in on purpose.**
  - The query is the FinanceBench question verbatim. Its instruction words are rare in 10-Ks, so BM25 weighs them heavily: "usd" is in 7 chunks, "question" in 16. A stop-word list drawn from these 57 questions would be tuning on the test set.
  - Questions say "FY2018"; filings say "2018". There's no rewriting.
- **How it's scored.**
  - One run scores the vector leg, the keyword leg and hybrid from the same searches. The vector leg's top 20 is checked against the committed vector results.
  - **Main result:** no-filter page hit@5, hybrid vs vector. Reported as questions fixed and broken, with an exact McNemar test and 95% Wilson intervals. Everything else is exploratory.
  - **Power:** vector misses 8 questions at hit@5. Reaching p < 0.05 needs a net gain of at least 6: 6, 7 or 8 fixed with none broken, or all 8 fixed with one broken. A smaller gain can't be told apart from chance. (Corrected after review: the first version said every case needed none broken.)
  - **Label misses:** changes on the 10 known label misses ([above](#where-vector-only-retrieval-misses): FinanceBench ids 00517, 00585, 00684, 00995, 01077, 01226, 01279, 01351, 02608, 10285) are reported separately.
- **Sensitivity check** (`--grid`): RRF k in {10, 30, 60, 100} × depth in {20, 50, 100}, plus the opposite tie-break. It is run after the main result, every cell is reported, and none replaces the settings above.
- **Already seen:** during design, the keyword top 10 for one question (3M FY2018 capex) was compared with its evidence page.

#### Results: hybrid is worse than vector alone

Real runs on 27 Oct 2026: the same 57 questions and gold labels, top 20.
- **Same baseline.** One run scores all three methods. The vector leg's top 20 is identical to the committed vector run in 171 of 171 searches, on the same HNSW build.
- **HNSW vs exact.** With exact vector search the metrics are the same to two decimals. The fused top 20 is the same chunks in the same order in 170 of 171 searches; the other is 00757, no filter.

| Setting | Method | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | MRR | All passages @5 / @20 |
|---|---|---|---|---|---|---|---|---|
| No filter | Vector | 53% (30) | 77% (44) | **86% (49)** | 95% (54) | 96% (55) | 0.67 | 4 / 9 of 13 |
| | Keyword (BM25) | 5% (3) | 9% (5) | 9% (5) | 14% (8) | 23% (13) | 0.08 | 0 / 0 |
| | Hybrid (RRF) | 23% (13) | 42% (24) | **53% (30)** | 82% (47) | 95% (54) | 0.38 | 2 / 7 |
| Company + year | Vector | 54% (31) | 82% (47) | 95% (54) | 96% (55) | 98% (56) | 0.70 | 4 / 9 |
| | Keyword (BM25) | 16% (9) | 23% (13) | 30% (17) | 42% (24) | 53% (30) | 0.22 | 0 / 1 |
| | Hybrid (RRF) | 30% (17) | 46% (26) | 53% (30) | 72% (41) | 95% (54) | 0.43 | 1 / 6 |

With the company filter, hit@5 is 86% for vector, 14% for keyword and 47% for hybrid.

- **Main result: hybrid loses 19 questions at hit@5.**
  - It fixes 2 and breaks 21 (exact McNemar p < 0.001). Hybrid is 53% (95% CI 40–65%) against vector's 86% (75–93%).
  - 4 of the 21 broken questions are known label misses (00517, 00995, 01351, 10285); no label miss is fixed. Without them it is 2 fixed and 17 broken (p = 0.0007).
  - The 2 fixed questions (Boeing 00464, Verizon 00215) had the evidence at vector rank 6; both legs found it.
- **Why the keyword leg is weak.**
  - With no filter, it has the evidence page nowhere in its top 50 for 35 of 57 questions; even inside the right filing, for 17. The vector leg: 1 and 0.
  - **Tables:** 52 of the 83 gold chunks are tables (40 of 57 questions have one). Only 28 of the keyword leg's 285 top-5 chunks are tables, against 171 for vector. Questions use analysts' words and statements use line items: "capital expenditure" against "Purchases of property, plant and equipment".
  - **Rare words in the question's wording** get the highest weights: "usd" is in 7 chunks, "question" in 16, "explain" in 8. "FY2018" is in none.
  - **No sense of company or year** beyond the words: with no filter, 135 of its 285 top-5 chunks are from the right company and 55 from the right filing. Vector: 285 and 220.
- **Why fusion makes it worse.**
  - RRF puts chunks found by both legs first. With no filter they take 217 of the 285 hybrid top-5 slots, and the vector leg's first hit drops out of the top 5 for 25 of 57 questions.
  - The design predicted that hit@1 and MRR would fall. It also predicted a rise in hit@20, which did not happen: hit@20 fell in every setting (no filter 55 → 54, company 55 → 51, company + year 56 → 54) and no question was fixed at hit@20.
  - For 19 of the 21 broken questions, the keyword leg never had the evidence page in its top 50, and the vector's hit (rank 1–5) fell to ranks 6–18.
    - In 15 of them, chunks the two legs shared pushed it down.
    - In the other 4 (00720, 00517, 05915, 00790), the legs share 0–2 chunks. With equal weights RRF then interleaves the two lists (vector 1, keyword 1, vector 2, …), so vector rank r falls to about 2r − 1, below the keyword leg's own off-target chunks.
- **The keyword leg adds no recall.** The evidence page is in either leg's top 50 for 56 of 57 questions with no filter, the same as the vector leg alone. With the company filter it's 57 against 56. A reranker over both lists would gain almost nothing from it.
- **By question type** (no filter, hit@5; vector / keyword / hybrid):
  - domain-relevant: 26 / 3 / 17 of 32;
  - metrics-generated: 15 / 0 / 9 of 15;
  - novel-generated: 8 / 2 / 4 of 10.
- **Sensitivity check** ([`results/retrieval_hybrid_grid.json`](results/retrieval_hybrid_grid.json), with the questions fixed and broken against vector for every cell): no setting beats vector alone, and the main run's cell is reproduced (53%). Hit@5 with no filter, against vector's 86% (49/57):
  - **20 per leg, any RRF k (10, 30, 60 or 100):** 79% (45/57), with 2 fixed and 6 broken (p = 0.29). That can't be told apart from vector, and 4 of the 6 broken are label misses (00517, 00995, 01226, 01351). It was found on the test set, so it is not adopted. With the right filing, the same cells lose clearly: 72–79% against 95%, with 0 fixed and 9–13 broken (p ≤ 0.004).
  - **50 or 100 per leg:** 44–77%. A smaller k helps at these depths, by letting each leg's own first hits beat weakly shared chunks.
  - **Keyword-first tie-breaks** (k 60, 50 per leg): 51%.
  - **Why fewer per leg helps:** far fewer chunks are shared. Over the 57 no-filter questions, the legs' top 20s share 108 chunks and their top 50s share 368.
- **Latency** ([`results/latency_hybrid.json`](results/latency_hybrid.json), `eval-retrieval --method hybrid --timing 3`). Median / p95 over 3 timed passes of all 171 searches after a warm-up, each leg at depth 50, legs one after the other, load average 2–5 on this 8 GB Mac:
  - no filter: vector 3 / 5 ms, keyword 119 / 324 ms, hybrid 124 / 328 ms;
  - company + year: vector 4 / 6 ms, keyword 42 / 120 ms, hybrid 47 / 125 ms.

  BM25 over the postings view is most of the cost. The single scored run (in the results files) was slower, under heavier load. The cost per question is unchanged: one query embedding, and BM25 runs in Postgres.
- **Decision:** vector search stays the default (`ask` uses it). Hybrid and keyword search stay available for comparison.
- **Next:**
  - Rerank the vector top 20–50.
  - Query decomposition for multi-statement questions.
  - Pull company and year out of the question.
  - Keyword search would need the question rewritten into filing vocabulary first (a job for the agents, days 9–12).
  - Any fusion tuning, such as 20 per leg, needs a development set outside the 57, e.g. questions on the 9 corpus filings no evaluation question uses.
- **Results:**
  - [`results/retrieval_hybrid.json`](results/retrieval_hybrid.json)
  - [`results/retrieval_hybrid_exact.json`](results/retrieval_hybrid_exact.json)

### Answer accuracy

`python -m finrag eval-answers` (`--estimate` counts the tokens and prices a run without any paid call). `python -m finrag answer "..."` answers one question, with its sources. Code: [`finrag/answer/`](finrag/answer), [`finrag/eval/answers.py`](finrag/eval/answers.py).

#### Decided before the first run

As for hybrid search, all of this was fixed before any answer to the 57 questions was generated, and it isn't changed after seeing results. A later change would be a new, labelled row.

- **Questions:** the same 57, asked verbatim.
- **Three settings**, mirroring the FinanceBench paper's shared store, single store and oracle:
  - **No filter:** the top 10 chunks from vector search over all 30 filings, the system as built. This is the main result.
  - **Company + year:** the top 10 within the right filing.
  - **Evidence pages:** every chunk on FinanceBench's evidence pages. This takes retrieval out and leaves reading and arithmetic.
- **Answers:** `gemini-3.8-flash`, with the model's default thinking, seed 0 and JSON output (the answer, the numbers of the sources it used, and whether the sources hold an answer).
  - The prompt ([`finrag/answer/generate.py`](finrag/answer/generate.py)) gives numbered sources with their citations, then the question.
  - It asks for the answer first, then the working, and forbids outside figures: if something is missing, the model must say what.
- **Grading:** `gemini-4-argon`, a larger model than the one answering, compares each answer with FinanceBench's reference answer and justification.
  - Labels: correct, incorrect or refusal.
  - Rules ([`finrag/eval/answers.py`](finrag/eval/answers.py)): figures must agree at the precision asked; rounding within about 1% and equivalent units are fine; extra detail is fine unless it contradicts the reference.
- **A check on the grader:** a second grader from another model family, a Claude subagent that doesn't see the Gemini grades, labels every answer too. Agreement and Cohen's kappa are reported, with the disagreements.
- **Reported:**
  - **Main result:** no-filter accuracy, as correct out of 57 with a 95% Wilson interval.
  - **Exploratory:** the other two settings and question-by-question comparisons between settings (exact McNemar), by question type, and by whether the evidence was in the context.
  - **Cost and latency:** from the API's own token counts, at the prices of 30 Oct 2026. Replies are cached, so a re-run costs nothing.
- **One run.** Gemini isn't fully deterministic even with a seed. There are no re-runs to pick the best.

#### Results: 95% correct with no filter

One real run on 31 Oct 2026: 171 answers (57 questions × 3 settings) and 171 grades. It cost **$1.55**, after a $0.005 smoke test on a question of my own.

| Setting | Correct (Gemini grader) | 95% CI | Correct, second grader | Correct by both graders | Incorrect | Refusals |
|---|---|---|---|---|---|---|
| No filter (top 10) | **54 / 57 (95%)** | 86–98% | 54 / 57 | 52 / 57 (91%) | 3 | 0 |
| Company + year (top 10) | 54 / 57 (95%) | 86–98% | 54 / 57 | 52 / 57 | 3 | 0 |
| Evidence pages | 56 / 57 (98%) | 91–100% | 55 / 57 | 55 / 57 | 1 | 0 |

- **The graders agree on 162 of 171 grades.** Cohen's kappa is only 0.37: with almost every answer correct, chance agreement is already high, so kappa is low even though raw agreement is 95%. The 9 disagreements fall on 5 questions, all judgement calls:
  - **PepsiCo restructuring (01328).** The question says to answer 0 if the income statement shows no line for it. The model said 0, then gave the $411 million. Gemini: correct; second grader: incorrect.
  - **Verizon capital intensity (00215).** "Yes", as in the reference, but without the reference's assets-to-revenue ratio. Gemini: incorrect; second grader: correct.
  - **Verizon quick ratio (00216).** The model computes the reference's 0.54, then argues the ratio doesn't suit a telecom company. Gemini: correct; second grader: incorrect.
  - **Boeing tax rate (00585).** The filing's own figures, (0.6)% and 14.7%, with the opposite sign convention to the reference.
  - **Best Buy gross margins (00685).** The right conclusion from FY2013–2019 data, because the FY2022–23 pages weren't retrieved.
  - The strictest count, correct by both graders, is in the table: 52 of 57 with no filter.
- **Retrieval is rarely the limit.** With no filter, the 10 chunks held an evidence page for 54 questions, and 52 of those were answered correctly. Of the 3 questions whose evidence page wasn't retrieved, 2 were still answered correctly (both Boeing) from other pages. One of them, the tax rate (00585), is a known label miss. The settings don't differ beyond chance: 1 question each way between no filter and company + year, and 0 against 2 between no filter and evidence pages (p = 0.5).
- **What's wrong** with the 3 incorrect no-filter answers (Gemini grader):
  - **CVS "capital intensive?" (00790):** "no" in all three settings, where the reference says "yes" on the strength of a low return on assets. The model reasoned from fixed assets.
  - **Best Buy (00685)** and **Verizon (00215):** as above.
- **By question type** (no filter): metrics-generated 15 / 15, novel-generated 10 / 10, domain-relevant 29 / 32.
- **Citations:** 51 of the 54 correct no-filter answers cite a chunk on an evidence page. The model never declared a question unanswerable.
- **Cost per question** (no filter):
  - **Answering:** $0.0077, from 4,789 input and 1,107 output tokens on average (783 of them thinking).
  - **Grading:** $0.0025 per answer, which is evaluation only.
- **Latency** (no filter, 4 calls at a time): answering takes 7.5 s median and 20.4 s p95. That dwarfs retrieval (about 0.3 s).
- **Caveats.**
  - 57 questions is a small test, as the interval of 86–98% shows.
  - FinanceBench has been public since 2023, so the models may have seen some of its questions. Nothing here rules that out.
  - The answers include their working (median 112 words). The graders score the conclusion and the figures, not the length.
- **Results:**
  - [`results/answers.json`](results/answers.json) (labels, citations, chunk ids, tokens, cost; no FinanceBench text and no answers);
  - [`results/answers_second_grader.json`](results/answers_second_grader.json) (both graders' labels).

### Multi-agent answers (Google ADK)

`python -m finrag agent "..."` answers one question and prints each step's tokens, cost and time. `python -m finrag eval-agents --max-usd 5` scores all 57. Code: [`finrag/agents/pipeline.py`](finrag/agents/pipeline.py), [`finrag/eval/agents.py`](finrag/eval/agents.py).

#### Decided before the first run

- **Pipeline:** one ADK `SequentialAgent`, with the steps passing results through session state.
  1. **Planner** (`gemini-3.1-flash-lite`, the cheap model, because it reads only the question): the company (from the 10 in the corpus), the fiscal years named, and 1–4 searches, one per figure or statement needed.
  2. **Retriever** (code, no model): the question's top 10 and each search's top 5 by vector search. They're filtered to the planned company and years (a year not in the corpus is dropped), merged without repeats, and capped at 20 chunks.
  3. **Analyst** (`gemini-3.8-flash`): answers from the numbered sources, as in the single-call version, but does every calculation with a `calculate` tool. That tool is a safe arithmetic evaluator, not Python `eval`.
  4. **Verifier** (`gemini-3.8-flash`): checks every figure against the sources, redoes every calculation with the tool, and checks that the conclusion answers the question. It returns the draft unchanged, or a corrected answer.
- **Scored like the single-call answers:** the same 57 questions with no filter given (the planner must find the company and year), the same grader (`gemini-4-argon`) and rubric. The analyst's draft is graded as well, so the verifier's effect is measured.
- **Reported:**
  - **Main result:** correct out of 57, compared question by question with the single call's no-filter row (54/57, exact McNemar). That row is already near its ceiling, so a gain of 3 at most is possible, and it couldn't reach significance.
  - **Also reported:** how often the planner gets the company and year right; whether an evidence page is in the sources; what the verifier changed, fixed and broke; and tokens, cost and time per step.
- **One run**, saved question by question to `data/eval/agents_runs.jsonl`. There's no reply cache here, so a re-run would pay again; the runner skips finished questions instead.

#### Results: no better than one call, at 8× the cost

One real run on 1 Nov 2026, after a smoke test on a question of my own: PepsiCo revenue growth, 8.70%, right and verified, $0.037.
- **Crash fix.** The run stopped at question 15 of 57: ADK once ended the analyst's turn with no structured draft, and the verifier's prompt then failed. Since then, the verifier is told when there's no draft and answers from the sources itself. This never happened again in the remaining 43 questions. All 57 ran with the same prompts and models.

| | Single call, no filter | Agents |
|---|---|---|
| Correct | **54 / 57 (95%)** | 52 / 57 (91%, 95% CI 81–96%) |
| Incorrect / refusals | 3 / 0 | 3 / 2 |
| Cost per question | $0.0077 | **$0.064** (8.3×) |
| Time per question, median / p95 | 7.5 / 20.4 s | **32.7 / 107.7 s** |

- **No difference beyond chance.** The agents get 1 question the single call misses (Verizon capital intensity, 00215), and miss 3 it gets (exact McNemar p = 0.625).
- **The planner works, and costs almost nothing.** `gemini-3.1-flash-lite` named the right company for 57 of 57 questions and included the right year for 55. The other two name no single filing year: Amcor's industry, and Boeing's forecast for FY2023, which sits in the FY2022 filing. It costs $0.0002 and 0.9 s per question.
- **The losses come from the retriever's year order.**
  - For "improving gross margin as of FY2023" (Amcor 00684) and "as of FY2022" (Boeing 00678), the planner asked for the named year plus earlier ones.
  - The retriever searches one year at a time, oldest first, then keeps the first 20 chunks. That cut the most recent year, the one asked about, so the analyst correctly refused: "the sources only cover through FY2022".
  - This is why an evidence page reached the sources for only 50 of 57 questions, against 54 for the single call's top 10.
  - The fixed method set only the cap and the question-first order, not the year order. Putting the named year first is the obvious fix; it would be a new, labelled row.
  - The third loss is PepsiCo 01328, the disputed "answer 0 or $411 million" question from the single-call run.
- **The verifier changed 1 answer of 57**, fixing none and breaking none. It costs $0.031 per question (it re-reads all the sources), almost half the total, and adds 14 s.
- **Where the cost and time go** (per question, median time):

  | Step | Time | Cost | Model calls | Calculator calls | Input tokens |
  |---|---|---|---|---|---|
  | Planner | 0.9 s | $0.0002 | 1 | – | 471 |
  | Retriever | 0.7 s | (search only) | – | – | 14.8 sources |
  | Analyst | 18.8 s | $0.033 | 3.6 | 4.0 | 33,066 |
  | Verifier | 14.2 s | $0.031 | 3.2 | 4.25 | 32,416 |

  Each calculator call is another model turn that re-sends the whole prompt, so the analyst reads about 33,000 tokens to answer from about 15 chunks.
- **What to keep:**
  - The planner's company and year: cheap, accurate, and it lifts retrieval (hit@5 from 86% to 95% when the right filing is used).
  - The calculator for questions that need arithmetic.
  - Drop the verifier, or keep it only for answers with calculations.
  - Put the named year first in retrieval.

  On these 57 questions, which one call already answers at 95%, the agents can't show a gain. That needs the harder test set.
- **Spend:** $3.65 for the agent runs, plus grading and one debug run of the failed question (about $0.40 together, estimated). So about $4.05 in all, under the $5 cap.
- **Results:** [`results/agents.json`](results/agents.json) (plans checked, chunk ids, labels for the draft and final answer, tokens, cost and time per step; no FinanceBench text and no answers).

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
