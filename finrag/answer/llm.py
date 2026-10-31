"""Gemini text generation with an on-disk cache and token and cost accounting.

Replies are cached in data/cache/replies.sqlite, keyed by model, settings and prompt, so a
re-run never pays twice. A cached reply keeps the tokens it cost when it was made.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# USD per 1M tokens, paid tier, standard (not batch), from
# https://ai.google.dev/gemini-api/docs/pricing ("Last updated 2026-10-30 UTC").
# Output includes thinking tokens.
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),  # until 31 Dec 2026; $1.50 / $7.50 from 1 Jan 2027
    "gemini-4-argon": (2.00, 10.00),  # until 30 Nov 2026; $4.00 / $20.00 from 1 Dec 2026
}


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD at the prices above."""
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


class Budget:
    """A spending cap shared by several models: calls stop once fresh replies have cost
    `limit` USD. Calls already in flight finish, so a run can overshoot by a few calls."""

    def __init__(self, limit: float | None = None):
        self.limit, self.spent = limit, 0.0
        self.lock = threading.Lock()

    def check(self) -> None:
        if self.limit is not None and self.spent >= self.limit:
            raise RuntimeError(f"Spending cap of ${self.limit:.2f} reached (${self.spent:.4f} "
                               "spent). Replies so far are cached; raise the cap to go on.")

    def add(self, usd: float) -> None:
        with self.lock:
            self.spent += usd


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int  # answer plus thinking, as billed
    thinking_tokens: int
    ms: float  # time of the API call that made it
    cached: bool = False

    @property
    def data(self) -> dict:
        """The reply parsed as JSON (replies use a response schema)."""
        return json.loads(self.text)


class ReplyCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # shared by worker threads, so one statement at a time
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT)")

    def get(self, key: str) -> Reply | None:
        with self.lock:
            row = self.db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        return Reply(**{**json.loads(row[0]), "cached": True}) if row else None

    def put(self, key: str, reply: Reply) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO replies (key, reply) VALUES (?, ?)",
                            (key, json.dumps({**asdict(reply), "cached": False})))
            self.db.commit()


class LLM:
    """One Gemini model with fixed settings: system instruction, JSON response schema,
    thinking level (None: the model's default) and seed."""

    def __init__(self, api_key: str, model: str, cache_dir: Path, system: str,
                 schema: dict, thinking_level: str | None = None, seed: int = 0,
                 budget: Budget | None = None):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model, self.system, self.schema = model, system, schema
        self.thinking_level, self.seed = thinking_level, seed
        self.cache = ReplyCache(cache_dir / "replies.sqlite")
        self.budget = budget or Budget()
        self.calls = 0

    def settings(self) -> dict:
        return {"model": self.model, "system": self.system, "schema": self.schema,
                "thinking_level": self.thinking_level, "seed": self.seed}

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(json.dumps([self.settings(), prompt], sort_keys=True).encode()
                              ).hexdigest()

    def _config(self):
        from google.genai import types

        thinking = (types.ThinkingConfig(thinking_level=self.thinking_level)
                    if self.thinking_level else None)
        return types.GenerateContentConfig(
            system_instruction=self.system, response_mime_type="application/json",
            response_json_schema=self.schema, thinking_config=thinking, seed=self.seed)

    def count_tokens(self, prompt: str) -> int:
        """Input tokens of a prompt plus the system instruction (a free API call)."""
        res = self.client.models.count_tokens(model=self.model,
                                              contents=f"{self.system}\n\n{prompt}")
        return res.total_tokens

    def generate(self, prompt: str) -> Reply:
        from google.genai import errors

        key = self._key(prompt)
        if (hit := self.cache.get(key)) is not None:
            return hit
        self.budget.check()
        for attempt in range(6):
            try:
                t0 = time.perf_counter()
                self.calls += 1
                res = self.client.models.generate_content(model=self.model, contents=prompt,
                                                          config=self._config())
                ms = 1000 * (time.perf_counter() - t0)
                break
            except errors.APIError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                log.warning("Gemini %s on %s, retrying", e.code, self.model)
                time.sleep(min(60, 2 ** (attempt + 2)))
        u = res.usage_metadata
        thinking = u.thoughts_token_count or 0
        reply = Reply(text=res.text or "", input_tokens=u.prompt_token_count or 0,
                      output_tokens=(u.candidates_token_count or 0) + thinking,
                      thinking_tokens=thinking, ms=round(ms, 1))
        self.budget.add(cost(self.model, reply.input_tokens, reply.output_tokens))
        json.loads(reply.text)  # a reply that isn't valid JSON is not cached
        self.cache.put(key, reply)
        return reply
