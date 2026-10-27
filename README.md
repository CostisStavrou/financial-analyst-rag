# Financial Analyst RAG

*Started 27 Oct 2026. Target: ready to show in the Google Forward Deployed Engineering Intern (BS/MS, 2027, London) application, which closes 13 Nov 2026.*

A multi-agent RAG system that answers questions about US companies' annual reports (10-K filings), with exact numbers, page citations and a cross-check against the SEC's structured financial data. Built on Google Cloud with Gemini.

## Why

- Annual reports are 100–300 pages each. Across dozens of companies and years, that's far beyond any chat context: retrieval is genuinely needed.
- It shows what the Google posting asks for: RAG pipelines, vector databases, multi-agent frameworks, LLM cost and latency metrics, shipping into a cloud environment.
- It is measurable: **FinanceBench** gives real questions with correct answers.

## Data

- **SEC EDGAR**: 10-K filings via the official free API. SEC rules: identify yourself in the `User-Agent` header (name + contact email) and stay under their request-rate limit.
- **SEC XBRL "company facts"**: structured financial figures, used to cross-check numbers.
- **FinanceBench**: question/answer set over 10-Ks (check its licence before use).
- Scope: about 10 companies × 3 years, chosen to overlap with FinanceBench questions.

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
