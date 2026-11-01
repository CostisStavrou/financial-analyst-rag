"""Multi-agent answering on Google's Agent Development Kit (ADK).

One ADK SequentialAgent, with session state passed between the steps:

1. planner (gemini-3.1-flash-lite, the cheap model: it reads only the question): which company
   and fiscal years the question is about, and 1-4 searches, one per figure or statement it
   needs;
2. retriever (code, no model; runs before the analyst): vector search for the question and for
   each planned search, filtered to the planned company and years, merged without repeats;
3. analyst (gemini-3.8-flash): answers from the numbered sources, with a calculator tool for
   arithmetic;
4. verifier (gemini-3.8-flash): checks every figure against the sources and re-does every
   calculation with the calculator; returns the draft unchanged or a corrected answer.

Token use and time are recorded per step from ADK's events and callbacks.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import psycopg
from pydantic import BaseModel, Field

from finrag.answer.llm import cost
from finrag.config import Company
from finrag.index.embed import Embedder
from finrag.retrieve.vector import Hit, search

PLANNER_MODEL = "gemini-3.1-flash-lite"
ANALYST_MODEL = VERIFIER_MODEL = "gemini-3.8-flash"
K_QUESTION, K_SEARCH, MAX_SOURCES = 10, 5, 20
STEPS = ("planner", "retriever", "analyst", "verifier")


class Plan(BaseModel):
    ticker: str | None = Field(description="Ticker of the company asked about, from the list; "
                                           "null if none of them")
    fiscal_years: list[int] = Field(description="Fiscal years the question needs; empty if it "
                                                "names none")
    searches: list[str] = Field(description="1 to 4 short searches, one per figure, statement "
                                            "or topic needed")


class Answer(BaseModel):
    answer: str = Field(description="The answer, then the working behind it")
    citations: list[int] = Field(description="Numbers of the sources used")
    answerable: bool = Field(description="false if the sources don't hold the answer")


class Verdict(Answer):
    problems: list[str] = Field(description="Problems found in the draft; empty if none")


# --- calculator tool ---------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e9):
            raise ValueError("power too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("only numbers, + - * / ** %, parentheses, abs, round, min, max, sqrt")


def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression exactly as written, e.g. "(34229 - 31750) / 31750 * 100".

    Args:
        expression: numbers (no commas, units or currency signs), + - * / ** %, parentheses,
            and abs, round, min, max, sqrt.
    """
    try:
        # thousands separators (1,577) go; commas between arguments (round(x, 2)) stay
        plain = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression)
        value = _eval(ast.parse(plain, mode="eval").body)
        if abs(value) > 1e18:
            raise ValueError("result too large")
        return {"expression": expression, "result": round(value, 10)}
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError) as e:
        return {"expression": expression, "error": str(e)}


# --- retrieval ----------------------------------------------------------------------------

def sources_text(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{n}] {h.citation()}\n{h.content}" for n, h in enumerate(hits, 1))


def retrieve(conn: psycopg.Connection, embedder: Embedder, model: str, question: str,
             plan: Plan, companies: list[Company]) -> list[Hit]:
    """The question's top K_QUESTION and each search's top K_SEARCH, filtered to the planned
    company and years (a ticker or year not in the corpus is dropped), merged without repeats:
    question first, then the searches, at most MAX_SOURCES."""
    by_ticker = {c.ticker: c for c in companies}
    ticker = plan.ticker if plan.ticker in by_ticker else None
    years = [y for y in plan.fiscal_years if ticker and y in by_ticker[ticker].fiscal_years]
    queries = [(question, K_QUESTION)] + [(s, K_SEARCH) for s in plan.searches[:4]]
    found: dict[str, Hit] = {}
    for text, k in queries:
        vec = embedder.embed_query(text)
        for fy in years or [None]:
            for h in search(conn, vec, model, k=k, ticker=ticker, fiscal_year=fy):
                found.setdefault(h.id, h)
    return list(found.values())[:MAX_SOURCES]


# --- the agents ---------------------------------------------------------------------------

def _planner_instruction(companies: list[Company]) -> str:
    listing = "\n".join(f"- {c.name} ({c.ticker}): fiscal years "
                        f"{', '.join(map(str, c.fiscal_years))}" for c in companies)
    return f"""\
You plan the research for a question about a US company's 10-K filings. The filings available:
{listing}

Return the ticker of the company the question is about (null if none of these), the fiscal \
years it needs ("FY2022", "fiscal 2022" and "2022" all mean fiscal year 2022; a change "from \
FY2021 to FY2022" needs both; leave it empty if no year is named), and 1 to 4 short searches \
for the filings, one per figure, statement or topic the answer needs (for a ratio, one per \
statement its inputs come from, e.g. "income statement net revenue", "balance sheet total \
assets")."""


ANALYST_INSTRUCTION = """\
You answer questions about US public companies' annual reports (10-K filings) as a careful \
financial analyst. Use only the numbered sources below: excerpts of the filings, where tables \
are written as "| cell | cell |" rows.

- Answer the question directly first, then give the figures and the working behind it.
- Do every calculation with the calculate tool, and give the formula and the numbers you took \
from the sources. Follow any units or rounding the question asks for.
- List the numbers of the sources you used in "citations".
- If the sources don't hold what you need, set "answerable" to false and say what is missing. \
Never fill gaps with outside knowledge of the company's figures.

Sources:

"""

VERIFIER_INSTRUCTION = """\
You check a draft answer to a question about a company's 10-K filings before it is sent. \
Using only the numbered sources below:

- check that every figure in the draft appears in a source it cites, with the right year, \
units and sign;
- re-do every calculation with the calculate tool;
- check that the conclusion follows from the figures and answers the question asked.

If everything holds, return the draft's answer, citations and answerable flag unchanged, with \
no problems. Otherwise list the problems and return a corrected answer, still using only the \
sources. Don't add caveats or rewrite a draft that is right.

Sources:

"""


NO_DRAFT = ("(none: the analyst returned no draft. Answer the question yourself from the "
            "sources, and list this as a problem.)")


def _draft_text(draft: dict | None) -> str:
    """The analyst's draft for the verifier. ADK occasionally ends the analyst's turn without
    a structured reply; the verifier then answers from the sources itself."""
    return json.dumps(draft) if draft else NO_DRAFT


@dataclass
class Trace:
    """Per question: the steps' token use and time, the plan, the sources and both answers."""
    tokens: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0, 0]))
    ms: dict = field(default_factory=dict)
    tool_calls: dict = field(default_factory=lambda: defaultdict(int))
    sources: list[Hit] = field(default_factory=list)


class Pipeline:
    def __init__(self, conn: psycopg.Connection, embedder: Embedder, embed_model: str,
                 companies: list[Company], api_key: str):
        from google.adk.agents import LlmAgent, SequentialAgent
        from google.adk.runners import InMemoryRunner

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            os.environ["GOOGLE_API_KEY"] = api_key  # ADK's Gemini client reads the environment
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "false")
        self.conn, self.embedder, self.embed_model = conn, embedder, embed_model
        self.companies = companies
        self.traces: dict[str, Trace] = {}
        self._t0: dict[tuple[str, str], float] = {}

        def sources(ctx) -> str:
            return sources_text(self.traces[ctx.session.id].sources)

        self.agent = SequentialAgent(name="finrag", sub_agents=[
            LlmAgent(name="planner", model=PLANNER_MODEL, output_schema=Plan, output_key="plan",
                     instruction=_planner_instruction(companies), include_contents="default",
                     before_agent_callback=self._start, after_agent_callback=self._stop),
            LlmAgent(name="analyst", model=ANALYST_MODEL, output_schema=Answer,
                     output_key="draft", tools=[calculate], include_contents="none",
                     instruction=lambda ctx: ANALYST_INSTRUCTION + sources(ctx)
                     + f"\n\nQuestion: {ctx.state['question']}",
                     before_agent_callback=self._retrieve, after_agent_callback=self._stop),
            LlmAgent(name="verifier", model=VERIFIER_MODEL, output_schema=Verdict,
                     output_key="verdict", tools=[calculate], include_contents="none",
                     instruction=lambda ctx: VERIFIER_INSTRUCTION + sources(ctx)
                     + f"\n\nQuestion: {ctx.state['question']}\n\nDraft answer (JSON): "
                     + _draft_text(ctx.state.get("draft")),
                     before_agent_callback=self._start, after_agent_callback=self._stop),
        ])
        self.runner = InMemoryRunner(agent=self.agent, app_name="finrag")

    # callbacks: time each step; run retrieval before the analyst
    def _start(self, callback_context):
        self._t0[(callback_context.session.id, callback_context.agent_name)] = time.perf_counter()

    def _stop(self, callback_context):
        sid, name = callback_context.session.id, callback_context.agent_name
        self.traces[sid].ms[name] = round(1000 * (time.perf_counter()
                                                  - self._t0.pop((sid, name))), 1)

    def _retrieve(self, callback_context):
        sid = callback_context.session.id
        t0 = time.perf_counter()
        plan = Plan.model_validate(callback_context.state["plan"])
        self.traces[sid].sources = retrieve(self.conn, self.embedder, self.embed_model,
                                            callback_context.state["question"], plan,
                                            self.companies)
        self.traces[sid].ms["retriever"] = round(1000 * (time.perf_counter() - t0), 1)
        self._t0[(sid, "analyst")] = time.perf_counter()

    async def run(self, question: str) -> dict:
        """Answer one question; returns the plan, sources, draft, verdict and per-step usage."""
        from google.genai import types

        session = await self.runner.session_service.create_session(
            app_name="finrag", user_id="eval", state={"question": question})
        trace = self.traces[session.id] = Trace()
        t0 = time.perf_counter()
        async for ev in self.runner.run_async(
                user_id="eval", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=question)])):
            u = ev.usage_metadata
            if u and ev.author in STEPS:
                t = trace.tokens[ev.author]
                thinking = u.thoughts_token_count or 0
                t[0] += u.prompt_token_count or 0
                t[1] += (u.candidates_token_count or 0) + thinking
                t[2] += thinking
                t[3] += 1  # model calls
            for p in (ev.content.parts if ev.content and ev.content.parts else []):
                if p.function_call:
                    trace.tool_calls[ev.author] += 1
        total_ms = round(1000 * (time.perf_counter() - t0), 1)
        state = (await self.runner.session_service.get_session(
            app_name="finrag", user_id="eval", session_id=session.id)).state
        del self.traces[session.id]
        models = {"planner": PLANNER_MODEL, "analyst": ANALYST_MODEL, "verifier": VERIFIER_MODEL}
        steps = {name: {"input_tokens": t[0], "output_tokens": t[1], "thinking_tokens": t[2],
                        "model_calls": t[3], "tool_calls": trace.tool_calls.get(name, 0),
                        "usd": round(cost(models[name], t[0], t[1]), 6),
                        "ms": trace.ms.get(name)}
                 for name, t in trace.tokens.items()}
        steps["retriever"] = {"ms": trace.ms.get("retriever"), "sources": len(trace.sources)}
        return {"plan": state.get("plan"), "sources": trace.sources, "draft": state.get("draft"),
                "verdict": state.get("verdict"), "steps": steps, "ms": total_ms,
                "usd": round(sum(s.get("usd", 0) for s in steps.values()), 6)}
