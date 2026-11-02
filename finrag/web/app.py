"""A small web demo of the default system: vector search over all 30 filings, the top 10
chunks, one Gemini call. One page asks a question and shows the answer, the passages it
cites with links to the filings on SEC.gov, and what the question cost and how long it took.

Spending: fresh Gemini replies are capped per day (UTC), and a call is refused unless the
worst case (MAX_OUTPUT tokens out) still fits under the cap. Saved replies cost nothing and
are always served. Questions are answered one at a time; one that waits more than LOCK_WAIT
seconds gets "busy". Query embeddings (about $0.000006 each) are not counted.

Built for running locally. Before it faces the internet it needs per-visitor rate limits,
the day's spend kept outside the process (in Postgres), security headers, a request size
limit and a shorter Gemini retry budget; see docs/BUILD.md.
"""

from __future__ import annotations

import logging
import re
import textwrap
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from finrag.answer import generate
from finrag.answer.llm import LLM, Budget, Reply, cost
from finrag.retrieve.vector import Hit

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("index.html")
K = 10  # chunks given to the model, as in the evaluation
MAX_QUESTION = 300  # characters
MAX_OUTPUT = 4096  # tokens, thinking included; evaluation answers averaged 1,107
WORST_INPUT = 8000  # tokens assumed for the cap's worst case; evaluation averaged 4,789
LOCK_WAIT = 45  # seconds a question may wait for the one before it
EXAMPLES = [
    "What were Amcor's trade receivables at the end of fiscal 2020?",
    "How many people did PepsiCo employ at the end of 2022?",
    "What was Best Buy's revenue in fiscal 2023?",
]


class CapReached(RuntimeError):
    pass


class DailyBudget(Budget):
    """A Budget that starts again at midnight UTC, and refuses a call unless `reserve` (the
    worst case for one call) still fits under the limit."""

    def __init__(self, limit: float, reserve: float = 0.0,
                 today=lambda: datetime.now(UTC).date()):
        super().__init__(limit)
        self.reserve, self.today = reserve, today
        self.day: date = today()

    def _roll(self) -> None:
        with self.lock:
            if (d := self.today()) != self.day:
                self.day, self.spent = d, 0.0

    def check(self) -> None:
        self._roll()
        if self.spent + self.reserve > self.limit:
            raise CapReached(f"Daily cap of ${self.limit:.2f} reached")

    def add(self, usd: float) -> None:
        self._roll()
        super().add(usd)


class Question(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def tidy(cls, v: str) -> str:
        v = " ".join(v.split())
        if not 3 <= len(v) <= MAX_QUESTION:
            raise ValueError(f"questions need 3 to {MAX_QUESTION} characters")
        return v


def _pages(h: Hit) -> str:
    p = f"p. {h.page_start}" + (f"–{h.page_end}" if h.page_end != h.page_start else "")
    return p + (f" (printed {h.page_label})" if h.page_label else "")


def _source(n: int, h: Hit, cited: set[int]) -> dict:
    return {
        "n": n, "cited": n in cited, "company": h.company, "ticker": h.ticker,
        "fiscal_year": h.fiscal_year,
        "document": "10-K" if h.document == "10-K" else f"10-K {h.document}",
        "section": h.section, "pages": _pages(h), "table": h.chunk_type == "table",
        "url": h.source_url, "excerpt": textwrap.shorten(h.content, 360, placeholder=" …"),
    }


def _inline_refs(text: str, n: int) -> set[int]:
    """Source numbers the answer's text points to, as [2] or [2, 3]."""
    found = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text)
    return {int(k) for group in found for k in group.split(",") if 1 <= int(k) <= n}


def create_app(search, llm: LLM, budget: DailyBudget, companies: list[str] = (),
               examples: list[str] = EXAMPLES) -> FastAPI:
    """search(question) -> the top K hits; companies: what the page says it covers. Built
    from settings by `from_settings`; tests pass fakes."""
    app = FastAPI(title="finrag demo", docs_url=None, redoc_url=None, openapi_url=None)
    one_at_a_time = threading.Lock()

    @app.get("/", include_in_schema=False)
    async def page() -> FileResponse:
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/api/info")
    async def info() -> dict:
        budget._roll()
        return {"examples": examples, "companies": list(companies), "model": llm.model,
                "k": K, "spent_today_usd": round(budget.spent, 4),
                "cap_usd_per_day": budget.limit}

    @app.post("/api/answer")
    def answer(q: Question) -> dict:
        if not one_at_a_time.acquire(timeout=LOCK_WAIT):
            raise HTTPException(503, "The demo is busy with other questions. Try again in a "
                                     "minute.")
        try:
            t0 = time.perf_counter()
            hits = search(q.question)
            t1 = time.perf_counter()
            if hits:
                a, reply = generate.answer(llm, q.question, hits)
            else:  # nothing indexed for this model: say so rather than pay for a guess
                a = {"answer": "No passages were found to answer from.", "citations": [],
                     "answerable": False}
                reply = Reply("", 0, 0, 0, 0.0)
        except CapReached as e:
            raise HTTPException(429, f"Today's demo limit of ${budget.limit:.2f} in Gemini "
                                     "calls is used up. It resets at midnight UTC; questions "
                                     "answered before still work.") from e
        except Exception as e:
            log.exception("answer failed")
            raise HTTPException(502, "Something went wrong while answering. Try again in a "
                                     "minute.") from e
        finally:
            one_at_a_time.release()
        cited = set(a["citations"]) | _inline_refs(a["answer"], len(hits))
        return {
            "question": q.question, "answer": a["answer"], "answerable": a["answerable"],
            "sources": [_source(n, h, cited) for n, h in enumerate(hits, 1)],
            "model": llm.model, "cached": reply.cached,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "usd": round(cost(llm.model, reply.input_tokens, reply.output_tokens), 4),
            "search_ms": round(1000 * (t1 - t0)),
            "answer_ms": round(reply.ms),  # for a saved reply, the time it took when made
            "spent_today_usd": round(budget.spent, 4), "cap_usd_per_day": budget.limit,
        }

    return app


class _Database:
    """One connection, opened again if it drops. Used one request at a time."""

    def __init__(self, url: str):
        from finrag.index import db

        self.url, self.connect = url, db.connect
        self.conn = db.connect(url)

    def search(self, vec, model: str, k: int) -> list[Hit]:
        import psycopg

        from finrag.retrieve import vector

        try:
            return vector.search(self.conn, vec, model, k=k)
        except psycopg.OperationalError:
            log.warning("database connection lost; reconnecting")
            self.conn = self.connect(self.url)
            return vector.search(self.conn, vec, model, k=k)


def from_settings(settings, usd_per_day: float) -> FastAPI:
    from finrag.config import load_companies
    from finrag.index.embed import make_embedder

    database = _Database(settings.database_url)
    n = database.conn.execute(
        "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL AND embedding_model = %s",
        (settings.embed_model,)).fetchone()[0]
    if not n:
        raise RuntimeError(f"No chunks are embedded with {settings.embed_model}. Run "
                           "`finrag index` first.")
    # the embedder isn't safe across threads; searches run one at a time under the app's lock
    embedder = make_embedder(settings)
    budget = DailyBudget(usd_per_day,
                         reserve=cost(generate.MODEL, WORST_INPUT, MAX_OUTPUT))
    llm = generate.answerer(settings, budget=budget, max_output_tokens=MAX_OUTPUT)

    def search(question: str) -> list[Hit]:
        return database.search(embedder.embed_query(question), settings.embed_model, K)

    companies = [f"{c.name} ({', '.join(map(str, c.fiscal_years))})" for c in load_companies()]
    return create_app(search, llm, budget, companies)
