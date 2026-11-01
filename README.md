# Financial Analyst RAG

Ask a question about a US company's annual report (10-K) and get the answer with the filing and page it came from. Every design choice is measured on FinanceBench, for accuracy, cost and time.

Python · Gemini · Google Agent Development Kit (ADK) · Postgres + pgvector · SEC EDGAR · FinanceBench

## Results

Scored on the **57 FinanceBench questions** that fall inside the 30 filings indexed here (10 companies × 3 years), out of the 150 in FinanceBench's public sample. No filter is given: the system has to find the right company and year itself.

| Approach | Correct | 95% CI | Cost / question | Median time |
|---|---|---|---|---|
| **One Gemini call over vector search** (the default) | **54 / 57 (95%)** | 86–98% | **$0.0077** | **7.5 s** |
| Multi-agent pipeline on ADK (planner, retriever, analyst, verifier) | 52 / 57 (91%) | 81–96% | $0.064 | 32.7 s |

| Retrieval only (no LLM) | Evidence page in the top 5 | 95% CI |
|---|---|---|
| **Vector search** | **49 / 57 (86%)** | 75–93% |
| Hybrid: keyword (BM25) + vector, fused with RRF | 30 / 57 (53%) | 40–65% |

What the numbers say:

- **Use one call, not the agents.** The agents are no more accurate (1 question gained, 3 lost, exact McNemar p = 0.625), cost 8.3× as much and take 4× as long. Their verifier changed 1 answer in 57 and cost almost half the total.
- **Hybrid search was measured and rejected.** Keyword search misses the financial tables that hold most answers ("capital expenditure" in the question, "Purchases of property, plant and equipment" in the filing), and fusion pushes vector search's good hits down. Hit@5 fell from 86% to 53% (p < 0.001).
- **Worth keeping from the agents:** the planner. The cheapest Gemini model named the right company for 57 of 57 questions and included the right year for 55, at $0.0002 each. Searching only the right filing (given by hand, as an upper bound) lifts hit@5 from 86% to 95%; with the planner choosing, that gain is not shown yet (see Status).
- **The grading was checked.** A second grader from another model family (Claude) agreed on 162 of 171 grades. The strictest count, correct by both graders, is 52 / 57 for the single call. The agents were graded by Gemini only.
- **Caveats.** 57 questions is a small test, as the intervals show. FinanceBench has been public since 2023, so the models may have seen some of it.

The hybrid, answer and agent methods were written down before their runs and not tuned on the 57 questions; the vector baseline and its gold labels were built on the same 57. Every number comes from a real run, saved in [`results/`](results); the full write-up, with error analysis, is in [docs/EVALUATION.md](docs/EVALUATION.md).

## How it works

```mermaid
flowchart LR
    A[SEC EDGAR<br/>30 10-K filings] --> B[Parse and chunk<br/>sections, pages, tables]
    B --> C[gemini-embedding-001<br/>12,103 chunks]
    C --> D[(Postgres + pgvector<br/>HNSW index)]
    Q[Question] --> E[Vector search<br/>top 10 chunks]
    D --> E
    E --> F[gemini-3.8-flash<br/>answer + page citations]
    Q -.-> P[ADK pipeline:<br/>planner → retriever →<br/>analyst + calculator → verifier]
    D -.-> P
```

- **Ingest:** downloads each 10-K from EDGAR, splits it into its 10-K items and pages, and chunks it (about 500 tokens of text, or one table with its caption). Every chunk keeps company, fiscal year, section, page and the SEC URL, so answers can cite them.
- **Index:** Gemini embeddings in Postgres with pgvector; BM25 keyword search in plain SQL for the hybrid comparison.
- **Answer:** the top 10 chunks go to Gemini with their citations; the reply is JSON with the answer and the sources it used. Replies are cached and every call's tokens and cost are recorded; evaluation runs take a spending cap (`--max-usd`).
- **Agents** (ADK `SequentialAgent`): a cheap planner picks the company, years and searches; code retrieves the chunks; an analyst answers using a safe `calculate` tool for arithmetic; a verifier re-checks figures and calculations.

More detail: [docs/BUILD.md](docs/BUILD.md).

## Try it

Needs Python 3.11+, Docker and a Gemini API key. Embedding the whole corpus costs about $0.55 on a paid key and takes about 4–5 minutes (estimated from the measured $0.53 and 250 s for 11,403–11,453 of the 12,103 chunks, at the model's launch price).

```bash
git clone <this repo> financial-analyst-rag && cd financial-analyst-rag
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # fill in SEC_USER_AGENT (name + email) and GEMINI_API_KEY
docker compose up -d    # Postgres 17 + pgvector on localhost:5433
```

Then build the index and ask questions:

```bash
.venv/bin/python -m finrag download   # 30 10-Ks from SEC EDGAR (~140 MB)
.venv/bin/python -m finrag parse      # sections, pages, chunks
.venv/bin/python -m finrag db-init    # schema and indexes
.venv/bin/python -m finrag index      # embed and load the chunks
.venv/bin/python -m finrag ask "Amcor trade receivables at the end of fiscal 2020" --ticker AMCR --year 2020
.venv/bin/python -m finrag answer "How many people did PepsiCo employ at the end of 2022?"
.venv/bin/python -m finrag agent "By what percentage did PepsiCo's net revenue grow from fiscal 2021 to fiscal 2022?"
```

`ask` shows the retrieved chunks, `answer` gives one Gemini answer with sources, and `agent` runs the multi-agent pipeline and prints each step's tokens, cost and time. Retrieval output from a real run (27 Oct, Amcor's FY2020 net receivables; FinanceBench's evidence is page 50):

```
   5 hits | embed 701 ms | search 16 ms
[1] score 0.782 | Amcor 10-K FY2020, Item 8. Financial Statements and Supplementary Data, p. 50 (printed page 50) | table
    ... | Trade receivables, net | 1,615.9 | 1,864.3 | ...
    https://www.sec.gov/Archives/edgar/data/1748790/000174879020000025/amcr-20200630.htm
```

The evaluations: `eval-retrieval` (add `--method hybrid` for hybrid search), `eval-answers` (`--estimate` prices a run first) and `eval-agents --max-usd 5 --tag <name>`. They download FinanceBench on first use (`.venv/bin/python -m finrag financebench` fetches it and shows the coverage). Tests: `.venv/bin/python -m pytest` (no API key needed).

## Status

Built and measured: ingestion, vector and hybrid search, single-call answers, the ADK pipeline, and the evaluation.

- **Agent retrieval fix, not yet re-measured.** The agents' two refusals came from how sources were capped at 20: the question's own top 10 came first for every planned year, so with two years none of the planner's searches got in, and with three the third year was cut too (in both refusals, the year asked about). The merge now takes each list's hits rank by rank, so every year and search gets in (tested in [`tests/test_agents.py`](tests/test_agents.py)). The 91% above is from the old version; a re-run will be reported as a separate row.
- **Not built yet:** a Cloud Run deployment, a cross-check of figures against the SEC's structured XBRL data, reranking, and a harder test set beyond these 57 questions.

## How this was built

Built from 27 Oct to 1 Nov 2026 with Claude Code (Anthropic) as the coding assistant, which is why commits carry its co-author line. The hybrid, answer and agent methods were written down before their runs, every paid run had an agreed spending cap, and every number above comes from a real run saved in [`results/`](results).

## Repo layout

```
config/companies.toml   corpus: companies, CIKs, fiscal years
finrag/ingest/          EDGAR download, HTML parsing, chunking
finrag/index/           embeddings (Gemini or local), Postgres schema and loading
finrag/retrieve/        vector search with filters, BM25 keyword search, hybrid (RRF)
finrag/answer/          Gemini answers with citations, reply cache, token and cost accounting
finrag/agents/          ADK pipeline: planner, retriever, analyst with calculator, verifier
finrag/eval/            FinanceBench loading, page check, retrieval, answer and agent scoring
finrag/cli.py           python -m finrag <command>
results/                evaluation results: per-question ranks, labels, tokens, cost; no FinanceBench text
docs/                   build details and the full evaluation write-up
tests/                  unit tests, no API calls; BM25 tests also run against the local database when it is up
```

## Data and licence

- **SEC EDGAR:** 10-K filings, fetched through SEC's free API with an identifying `User-Agent` and at most 2 requests a second. Filings are downloaded into `data/` and not committed.
- **FinanceBench:** [`PatronusAI/financebench`](https://huggingface.co/datasets/PatronusAI/financebench) (Islam et al., 2023, [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)), licensed **CC BY-NC 4.0**. It is downloaded on demand and not redistributed: result files and tests hold question ids, never question or answer text, and the docs refer to questions by id, quoting at most a short phrase.
- **Code:** [MIT](LICENSE).
