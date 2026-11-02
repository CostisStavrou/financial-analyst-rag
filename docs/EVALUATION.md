# Evaluation

The full write-up behind the [README's results](../README.md#results). Every number comes from a real run; result files are in [`results/`](../results). FinanceBench question ids are given so the questions can be looked up in the dataset; their text is not reproduced beyond a short phrase.

## Summary

FinanceBench's 57 questions on the 30 indexed filings, no filter given:

| Version | Evidence in top 5 | Answers correct | Cost / question | Median time |
|---|---|---|---|---|
| Vector only, one `gemini-3.8-flash` call over the top 10 | **86%** (49/57; CI 75–93%) | **95%** (54/57; CI 86–98%). Strict, both graders: 91% (52/57) ([results](#results-95-correct-with-no-filter)) | $0.0077 (answer; retrieval $0.000006) | 279 ms embed + 3 ms search + 7.5 s answer ¹ |
| Hybrid (BM25 + vector, RRF) | 53% (30/57; CI 40–65%) ([why lower](#results-hybrid-is-worse-than-vector-alone)) | – | $0.000006 (query embedding only) | 279 ms embed + 124 ms search ¹ |
| Agents (ADK: planner, retriever, analyst + calculator, verifier) | – (an evidence page reached the sources for 50/57; 54/57 with the fixed merge, [replayed offline](#replay-with-the-fixed-merge-no-api-calls)) | 91% (52/57; CI 81–96%), Gemini grader only ([results](#results-no-better-than-one-call-at-8-the-cost)) | $0.064 | 32.7 s, end to end |

CI: 95% Wilson interval. ¹ Medians from separate runs: query embedding from [`retrieval_vector_before_reindex.json`](../results/retrieval_vector_before_reindex.json) (later runs read the embedding cache), vector search from [`retrieval_vector.json`](../results/retrieval_vector.json), hybrid search from [`latency_hybrid.json`](../results/latency_hybrid.json), the answer call from [`answers.json`](../results/answers.json) (4 calls at a time). The agents' 32.7 s includes retrieval and every step, one question at a time.

## Retrieval (no LLM): is the evidence in the top k?

`python -m finrag eval-retrieval` (`--exact` for brute-force search). Real runs on 27 Oct 2026: 57 questions, the FinanceBench question verbatim as the query, top 20 chunks. Code: [`finrag/eval/retrieval.py`](../finrag/eval/retrieval.py).

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
  - [`results/retrieval_vector.json`](../results/retrieval_vector.json)
  - [`results/retrieval_vector_exact.json`](../results/retrieval_vector_exact.json)
  - [`results/retrieval_vector_before_reindex.json`](../results/retrieval_vector_before_reindex.json) (the first run, before the HNSW rebuild below)

### HNSW vs exact search

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

### Where vector-only retrieval misses

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

**Ideas this suggests:**

- Query decomposition (one sub-query per statement) for ratio questions: 3M's and CVS's income statements and Verizon's revenue table are not in the top 20.
- Reranking the top 20, or smaller chunks, for sentences buried in long chunks (Boeing customers).
- Extracting the company and year from the question: the right filing lifts hit@5 from 86% to 95%. (Measured later with the agents' planner choosing: 51 of 57, and the evidence in the top 10 for the same 54 questions; [see below](#replay-with-the-fixed-merge-no-api-calls).)

## Hybrid search: keyword + vector

`python -m finrag eval-retrieval --method hybrid` (`ask --method keyword|hybrid` for single questions). Code: [`finrag/retrieve/keyword.py`](../finrag/retrieve/keyword.py), [`finrag/retrieve/hybrid.py`](../finrag/retrieve/hybrid.py).

### Decided before the first run

The 57 questions are the only evaluation set, so every choice below was fixed, after three independent reviews of the design (separate Claude sessions), before any keyword or hybrid result was seen. Nothing is tuned on the 57. A change made after seeing results would be a new, labelled row.

- **Keyword leg: BM25 in plain SQL.** No extension beyond pgvector, so it should run unchanged on Cloud SQL and AlloyDB (not tested).
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
  - **Power:** vector misses 8 questions at hit@5. Reaching p < 0.05 needs a net gain of at least 6: 6, 7 or 8 fixed with none broken, or all 8 fixed with one broken. A smaller gain can't be told apart from chance.
  - **Label misses:** changes on the 10 known label misses ([above](#where-vector-only-retrieval-misses): FinanceBench ids 00517, 00585, 00684, 00995, 01077, 01226, 01279, 01351, 02608, 10285) are reported separately.
- **Sensitivity check** (`--grid`): RRF k in {10, 30, 60, 100} × depth in {20, 50, 100}, plus the opposite tie-break. It is run after the main result, every cell is reported, and none replaces the settings above.
- **Already seen:** during design, the keyword top 10 for one question (3M FY2018 capex) was compared with its evidence page.

### Results: hybrid is worse than vector alone

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
- **Sensitivity check** ([`results/retrieval_hybrid_grid.json`](../results/retrieval_hybrid_grid.json), with the questions fixed and broken against vector for every cell): no setting beats vector alone, and the main run's cell is reproduced (53%). Hit@5 with no filter, against vector's 86% (49/57):
  - **20 per leg, any RRF k (10, 30, 60 or 100):** 79% (45/57), with 2 fixed and 6 broken (p = 0.29). That can't be told apart from vector, and 4 of the 6 broken are label misses (00517, 00995, 01226, 01351). It was found on the test set, so it is not adopted. With the right filing, the same cells lose clearly: 72–79% against 95%, with 0 fixed and 9–13 broken (p ≤ 0.004).
  - **50 or 100 per leg:** 44–77%. A smaller k helps at these depths, by letting each leg's own first hits beat weakly shared chunks.
  - **Keyword-first tie-breaks** (k 60, 50 per leg): 51%.
  - **Why fewer per leg helps:** far fewer chunks are shared. Over the 57 no-filter questions, the legs' top 20s share 108 chunks and their top 50s share 368.
- **Latency** ([`results/latency_hybrid.json`](../results/latency_hybrid.json), `eval-retrieval --method hybrid --timing 3`). Median / p95 over 3 timed passes of all 171 searches after a warm-up, each leg at depth 50, legs one after the other, load average 2–5 on this 8 GB Mac:
  - no filter: vector 3 / 5 ms, keyword 119 / 324 ms, hybrid 124 / 328 ms;
  - company + year: vector 4 / 6 ms, keyword 42 / 120 ms, hybrid 47 / 125 ms.

  BM25 over the postings view is most of the cost. The single scored run (in the results files) was slower, under heavier load. The cost per question is unchanged: one query embedding, and BM25 runs in Postgres.
- **Decision:** vector search stays the default (`ask` uses it). Hybrid and keyword search stay available for comparison.
- **Ideas this suggests:**
  - Rerank the vector top 20–50.
  - Query decomposition for multi-statement questions.
  - Pull company and year out of the question.
  - Keyword search would need the question rewritten into filing vocabulary first (a job for an agent step).
  - Any fusion tuning, such as 20 per leg, needs a development set outside the 57, e.g. questions on the 9 corpus filings no evaluation question uses.
- **Results:**
  - [`results/retrieval_hybrid.json`](../results/retrieval_hybrid.json)
  - [`results/retrieval_hybrid_exact.json`](../results/retrieval_hybrid_exact.json)

## Answer accuracy

`python -m finrag eval-answers` (`--estimate` counts the tokens and prices a run without any generation call; uncached query embeddings are still made). `python -m finrag answer "..."` answers one question, with its sources. Code: [`finrag/answer/`](../finrag/answer), [`finrag/eval/answers.py`](../finrag/eval/answers.py).

### Decided before the first run

As for hybrid search, all of this was fixed before any answer to the 57 questions was generated, and it isn't changed after seeing results. A later change would be a new, labelled row.

- **Questions:** the same 57, asked verbatim.
- **Three settings**, mirroring the FinanceBench paper's shared store, single store and oracle:
  - **No filter:** the top 10 chunks from vector search over all 30 filings, the system as built. This is the main result.
  - **Company + year:** the top 10 within the right filing.
  - **Evidence pages:** every chunk on FinanceBench's evidence pages. This takes retrieval out and leaves reading and arithmetic.
- **Answers:** `gemini-3.8-flash`, with the model's default thinking, seed 0 and JSON output (the answer, the numbers of the sources it used, and whether the sources hold an answer).
  - The prompt ([`finrag/answer/generate.py`](../finrag/answer/generate.py)) gives numbered sources with their citations, then the question.
  - It asks for the answer first, then the working, and forbids outside figures: if something is missing, the model must say what.
- **Grading:** `gemini-4-argon`, a larger model than the one answering, compares each answer with FinanceBench's reference answer and justification.
  - Labels: correct, incorrect or refusal.
  - Rules ([`finrag/eval/answers.py`](../finrag/eval/answers.py)): figures must agree at the precision asked; rounding within about 1% and equivalent units are fine; extra detail is fine unless it contradicts the reference.
- **A check on the grader:** a second grader from another model family, a Claude subagent that doesn't see the Gemini grades, labels every answer too. Agreement and Cohen's kappa are reported, with the disagreements.
- **Reported:**
  - **Main result:** no-filter accuracy, as correct out of 57 with a 95% Wilson interval.
  - **Exploratory:** the other two settings and question-by-question comparisons between settings (exact McNemar), by question type, and by whether the evidence was in the context.
  - **Cost and latency:** from the API's own token counts, at the prices of 30 Oct 2026. Replies are cached, so a re-run costs nothing.
- **One run.** Gemini isn't fully deterministic even with a seed. There are no re-runs to pick the best.

### Results: 95% correct with no filter

One real run on 31 Oct 2026: 171 answers (57 questions × 3 settings) and 171 grades. It cost **$1.54** ($1.55 counting replies cached from earlier tests), after a $0.005 smoke test on a question of my own.

| Setting | Correct (Gemini grader) | 95% CI | Correct, second grader | Correct by both graders | Incorrect | Refusals |
|---|---|---|---|---|---|---|
| No filter (top 10) | **54 / 57 (95%)** | 86–98% | 54 / 57 | 52 / 57 (91%) | 3 | 0 |
| Company + year (top 10) | 54 / 57 (95%) | 86–98% | 54 / 57 | 52 / 57 | 3 | 0 |
| Evidence pages | 56 / 57 (98%) | 91–99.7% | 55 / 57 | 55 / 57 | 1 | 0 |

- **The graders agree on 162 of 171 grades.** Cohen's kappa is only 0.37: with almost every answer correct, chance agreement is already high, so kappa is low even though raw agreement is 95%. The 9 disagreements fall on 5 questions, all judgement calls:
  - **PepsiCo restructuring (01328).** The question says to answer 0 if the income statement shows no line for it. The model said 0, then gave the $411 million. Gemini: correct; second grader: incorrect.
  - **Verizon capital intensity (00215).** "Yes", as in the reference, but without the reference's assets-to-revenue ratio. Gemini: incorrect; second grader: correct.
  - **Verizon quick ratio (00216).** The model computes the reference's 0.54, then argues the ratio doesn't suit a telecom company. Gemini: correct; second grader: incorrect.
  - **Boeing tax rate (00585).** The filing's own figures, (0.6)% and 14.7%, with the opposite sign convention to the reference.
  - **Best Buy gross margins (00685).** The right conclusion from FY2013–2019 data, because the FY2022–23 pages weren't retrieved.
  - The strictest count, correct by both graders, is in the table: 52 of 57 with no filter.
- **Retrieval is rarely the limit.** With no filter, the 10 chunks held an evidence page for 54 questions, and 52 of those were answered correctly. Of the 3 questions whose evidence page wasn't retrieved, 2 were still answered correctly (both Boeing) from other pages. One of them, the tax rate (00585), is a known label miss. The settings don't differ beyond chance: 1 question each way between no filter and company + year, and 0 against 2 between no filter and evidence pages (p = 0.5). So on these 57 questions the right filing, given by hand, lifts retrieval (hit@5 86% → 95%) but not answers: 54 either way.
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
  - FinanceBench has been public since 2023, so the models may have seen some of its questions. Nothing here rules that out; answering with no sources at all would test it, and hasn't been run.
  - **Against the paper (indicative only).** The [FinanceBench paper](https://arxiv.org/abs/2311.11944) (Islam et al., 2023, Table 2) scored GPT-4-Turbo on all 150 questions: 19% correct with one vector store over all 360 filings, 50% with a store for the right filing, 85% given the evidence pages. The three settings here follow those: 95%, 95% and 98%, on 57 of the 150. The no-filter setting is far easier than the paper's shared store (30 filings, not 360, and every top-20 chunk came from the right company), so the closer comparisons are the other two. Even those differ in the model (much newer here), the grading (the paper's answers were labelled by people, these by a Gemini grader), the retrieval stack and prompts, and the questions (10-K only, 57 of the sample's 112 10-K questions, from roughly the 10 companies with the most of them, [docs/BUILD.md](BUILD.md)). How much each explains isn't measured.
  - There is no development set: the gold labels, the chunking and the HNSW fix were all worked out on these 57 questions.
  - The answers include their working (median 112 words). The graders score the conclusion and the figures, not the length.
- **Results:**
  - [`results/answers.json`](../results/answers.json) (labels, citations, chunk ids, tokens, cost; no FinanceBench text and no answers);
  - [`results/answers_second_grader.json`](../results/answers_second_grader.json) (both graders' labels). The second grading was a one-off cross-check by a Claude session given the same rubric and no Gemini labels; there is no command that re-runs it. Agreement on the main (no-filter) setting alone: 53 of 57, kappa 0.30.

## Multi-agent answers (Google ADK)

`python -m finrag agent "..."` answers one question and prints each step's tokens, cost and time. `python -m finrag eval-agents --max-usd 5 --tag <name>` scores all 57 as a new, named run; it refuses to resume from runs made with another retrieval merge. Code: [`finrag/agents/pipeline.py`](../finrag/agents/pipeline.py), [`finrag/eval/agents.py`](../finrag/eval/agents.py).

### Decided before the first run

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

### Results: no better than one call, at 8× the cost

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
  - The retriever searched one year at a time, in the order the planner listed them (oldest first in both cases), then kept the first 20 chunks. That cut the most recent year, the one asked about, so the analyst correctly refused: "the sources only cover through FY2022".
  - This is why an evidence page reached the sources for only 50 of 57 questions, against 54 for the single call's top 10.
  - More precisely, the question's own top 10 came first for every planned year: with two years they alone filled the cap of 20, so none of the planner's searches reached the analyst; with three, the third year was cut as well.
  - **Fixed in code on 1 Nov, after this run:** the ranked lists (one per search and year) are now merged rank by rank, so every year and search gets in under the cap ([`tests/test_agents.py`](../tests/test_agents.py)). Retrieval with the fix was replayed offline ([below](#replay-with-the-fixed-merge-no-api-calls)); answers weren't re-generated, so a paid re-run would be a new, labelled row.
  - The third loss is PepsiCo 01328, the disputed "answer 0 or $411 million" question from the single-call run.
- **The verifier changed 1 answer of 57**, fixing none and breaking none. It costs $0.031 per question (it re-reads all the sources), almost half the total, and adds 14 s.
- **Where the cost and time go** (per question: median time; mean cost, calls and tokens):

  | Step | Time | Cost | Model calls | Calculator calls | Input tokens |
  |---|---|---|---|---|---|
  | Planner | 0.9 s | $0.0002 | 1 | – | 471 |
  | Retriever | 0.7 s | (search only) | – | – | 14.8 sources |
  | Analyst | 18.8 s | $0.033 | 3.6 | 4.0 | 33,066 |
  | Verifier | 14.2 s | $0.031 | 3.2 | 4.25 | 32,416 |

  Each calculator call is another model turn that re-sends the whole prompt, so the analyst reads about 33,000 tokens to answer from about 15 chunks.
- **What to keep:**
  - The planner's company and year are cheap and accurate, but on these 57 questions they are unlikely to help the single call: replayed as a filter, the evidence is in its top 10 for the same 54 questions ([below](#replay-with-the-fixed-merge-no-api-calls)). No answer run with the filter has been done.
  - The calculator for questions that need arithmetic.
  - Drop the verifier, or keep it only for answers with calculations.
  - Merge retrieval so every planned year and search gets in (done in code; the replay below finds the evidence in the sources for 54 of 57). Under the cap of 20, each ranked list then keeps about 20 / (number of lists) hits, so with two years and four searches (ten lists) the question keeps only its top 2 per year. In the replay that cost one question its evidence page (00720: two years and two searches, so six lists of 3 or 4 hits each).

  On these 57 questions, which one call already answers at 95%, the agents can't show a gain. That needs the harder test set.
- **Spend:** $3.65 for the agent runs, plus grading and one debug run of the failed question (not metered; about $0.40 together, an estimate). So about $4.05 in all, under the $5 cap.
- **Results:** [`results/agents.json`](../results/agents.json) (plans checked, chunk ids, labels for the draft and final answer, tokens, cost and time per step; no FinanceBench text and no answers).

### Replay with the fixed merge (no API calls)

`python -m finrag eval-replay`, run on 2 Nov 2026. It takes the 57 plans saved by the agent run above (`data/eval/agents_runs.jsonl`) and runs today's retrieval on them. Every query embedding comes from the cache (225 of them), and the embedder refuses to call the API, so it cost nothing. Only retrieval is replayed: no answer is generated or graded, so these are evidence counts, not accuracy. It needs the run's local files and the embedding cache, which aren't in the repo, so it can't be re-run from a fresh clone.

The old merge isn't replayed alongside, so index drift since 1 Nov could in principle explain part of the change. It is unlikely: the replay's unfiltered question searches find the evidence in the top 5 and top 10 for the same 49 and 54 questions as [`retrieval_vector.json`](../results/retrieval_vector.json). Code: [`finrag/eval/replay.py`](../finrag/eval/replay.py).

| | Recorded run (old merge) | Replay |
|---|---|---|
| Agent sources hold an evidence page | 50 / 57 | **54 / 57** (5 gained, 1 lost; p = 0.22) |

- **Gained:** both refusals (Amcor 00684, Boeing 00678), plus 00723, 00685 and 00464, all with three planned years.
- **Lost:** 00720 (American Express, two planned years, two searches). With six lists under the cap of 20, each keeps only its top 3 or 4, and the evidence wasn't among them.

**The planner as a filter in front of the single call.** The single call's question (top 10, what the answer reads) searched within the planned company, and also the planned year when exactly one is left after dropping years not in the corpus, as `retrieve()` does (38 questions; company only for the other 19):

| Evidence page in | No filter | Planner's filter |
|---|---|---|
| Top 5 | 49 / 57 | 51 / 57 (2 gained, 0 lost; p = 0.5) |
| Top 10 | 54 / 57 | 54 / 57 (no change) |

The filter changes which chunks the single call reads (for a filtered year, only that filing's), but the evidence is in its top 10 for the same 54 questions either way. With the right filing given by hand the answers didn't move either (54 / 57 both ways, [above](#results-95-correct-with-no-filter)), so a paid answer run with the filter is unlikely to show a gain on these 57; none has been done.

- **Results:** [`results/replay_retrieval.json`](../results/replay_retrieval.json) (per question: planned years, the filter used, sources, and each evidence check; no FinanceBench text).

## Reproducing

The hybrid, answer and agent result files record the code they ran with as "the commit before, plus uncommitted changes", because the code was committed right after each run; the vector files record no code version. The commit that holds each one's code:

| Result files | Code in commit |
|---|---|
| `retrieval_vector.json`, `retrieval_vector_exact.json` | `90c3017` (committed together with these files) |
| `retrieval_vector_before_reindex.json` | the evaluation code of `90c3017`, but run before that commit's index fixes; the HNSW graph it measured no longer exists |
| `retrieval_hybrid*.json`, `latency_hybrid.json` | `5ceebe1` |
| `answers.json`, `answers_second_grader.json` | `6b6009e` |
| `agents.json` | `2375a8a` (the retrieval merge and the calculator's comma handling were changed after it) |
| `replay_retrieval.json` | the commit after `abb9b14`, which added the replay |

Generation is not fully deterministic even with a seed, so a fresh run could give different answers; how many would change isn't measured, since each setting was run once. Replies are cached in `data/cache/`, so re-running the same version on the same machine costs nothing.
