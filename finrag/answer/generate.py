"""Answer a question from retrieved 10-K chunks with Gemini, citing the chunks it used."""

from __future__ import annotations

from finrag.answer.llm import LLM, Budget, Reply
from finrag.retrieve.vector import Hit

MODEL = "gemini-3.8-flash"

SYSTEM = """\
You answer questions about US public companies' annual reports (10-K filings) as a careful \
financial analyst. Use only the numbered sources given with the question: excerpts of the \
filings, where tables are written as "| cell | cell |" rows.

- Answer the question directly first, then give the figures and the working behind it.
- For a calculation, give the formula and the numbers you took from the sources, and follow \
any units or rounding the question asks for.
- List the numbers of the sources you used in "citations".
- If the sources don't hold what you need, set "answerable" to false and say what is missing. \
Never fill gaps with outside knowledge of the company's figures."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The answer, then the working behind it."},
        "citations": {"type": "array", "items": {"type": "integer"},
                      "description": "Numbers of the sources used."},
        "answerable": {"type": "boolean",
                       "description": "false if the sources don't hold the answer."},
    },
    "required": ["answer", "citations", "answerable"],
}


def answerer(settings, model: str = MODEL, budget: Budget | None = None,
             max_output_tokens: int | None = None) -> LLM:
    return LLM(settings.gemini_api_key, model, settings.cache_dir, SYSTEM, SCHEMA,
               budget=budget, max_output_tokens=max_output_tokens)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Numbered sources with their citation, then the question (last, as Google advises for
    prompts with long context)."""
    sources = "\n\n".join(f"[{n}] {h.citation()}\n{h.content}" for n, h in enumerate(hits, 1))
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def parse(reply: Reply, n_sources: int) -> dict:
    """The reply's fields, keeping only citations that name a given source."""
    d = reply.data
    cited = [c for c in d.get("citations", []) if isinstance(c, int) and 1 <= c <= n_sources]
    return {"answer": str(d.get("answer", "")), "citations": sorted(set(cited)),
            "answerable": bool(d.get("answerable", True))}


def answer(llm: LLM, question: str, hits: list[Hit]) -> tuple[dict, Reply]:
    reply = llm.generate(build_prompt(question, hits))
    return parse(reply, len(hits)), reply
