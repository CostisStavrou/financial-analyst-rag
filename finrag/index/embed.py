"""Embedding providers with an on-disk cache.

- gemini: `gemini-embedding-001` through the Gemini API (google-genai SDK). Document and
  query vectors use the RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types. Below 3072 dims the
  vectors are not unit length, so they are normalised here.
- local:  BAAI/bge-base-en-v1.5 through fastembed (ONNX, CPU), 768-dim. For running the
  pipeline offline or without an API key; not the main model.

Vectors are cached in data/cache/embeddings.sqlite keyed by (model, dim, task, text), so
rebuilding the database never pays for the same embedding twice.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from collections import deque
from pathlib import Path

import numpy as np

from finrag.ingest.chunk import est_tokens

log = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    @staticmethod
    def key(model: str, dim: int, task: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{dim}|{task}|{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})"
            for k, blob in self.db.execute(q, part):
                out[k] = np.frombuffer(blob, dtype=np.float32)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            [(k, v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.db.commit()


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class RateLimited(Exception):
    def __init__(self, wait: float, quota: str):
        super().__init__(quota)
        self.wait, self.quota = wait, quota


class GeminiEmbedder:
    """Batches texts per request. Quotas depend on the key's tier and are only shown in AI
    Studio, so the batch size adapts: halved after each 429, after waiting the delay the API
    suggests. A daily quota stops the run (progress so far is cached).

    Optional pacing (EMBED_RPM texts/min, EMBED_TPM estimated tokens/min) keeps a free-tier
    key under its per-minute limits instead of bouncing off them; each text in a batch
    counts towards the request quota.
    """

    max_batch = 100  # texts per request
    max_batch_tokens = 20_000  # estimated tokens per request
    max_consecutive_429 = 20

    def __init__(self, api_key: str, model: str, dim: int, rpm: int = 0, tpm: int = 0):
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env (see .env.example), "
                "or set EMBED_PROVIDER=local to embed offline."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model, self.dim = model, dim
        self.tokens_sent = 0
        self.requests = 0
        self.rate_limited = 0
        self.rpm, self.tpm = rpm, tpm
        if rpm:
            self.max_batch = min(self.max_batch, rpm)
        if tpm:
            self.max_batch_tokens = min(self.max_batch_tokens, tpm)
        self._sent: deque[tuple[float, int, int]] = deque()  # (time, texts, tokens)

    def _pace(self, texts: int, tokens: int) -> None:
        """Sleep until this batch fits in the last-60-seconds budget."""
        if not (self.rpm or self.tpm):
            return
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0][0] >= 60:
                self._sent.popleft()
            used_t = sum(n for _, n, _ in self._sent)
            used_k = sum(k for _, _, k in self._sent)
            if (not self.rpm or used_t + texts <= self.rpm) and (
                not self.tpm or used_k + tokens <= self.tpm
            ):
                self._sent.append((now, texts, tokens))
                return
            time.sleep(max(0.5, 60 - (now - self._sent[0][0])))

    def _call(self, texts: list[str], task: str) -> list[np.ndarray]:
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        for attempt in range(6):
            try:
                self.requests += 1
                res = self.client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                self.tokens_sent += sum(est_tokens(t) for t in texts)
                return [_normalise(np.asarray(e.values, dtype=np.float32)) for e in res.embeddings]
            except errors.APIError as e:
                msg = str(e)
                if e.code == 429:
                    quota = ",".join(sorted(set(re.findall(r"quotaId['\"]?:\s*['\"]?([\w-]+)", msg))))
                    if "PerDay" in quota:
                        raise RuntimeError(
                            f"Gemini daily quota reached ({quota}). Embeddings so far are cached; "
                            "run `finrag index` again after the reset."
                        ) from e
                    hint = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
                    raise RateLimited(float(hint.group(1)) + 1 if hint else 60.0, quota) from e
                if e.code not in (500, 502, 503, 504) or attempt == 5:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str], task: str) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        i, strikes = 0, 0
        while i < len(texts):
            batch, tokens = [], 0
            for t in texts[i:]:
                n = est_tokens(t)
                if batch and (len(batch) >= self.max_batch or tokens + n > self.max_batch_tokens):
                    break
                batch.append(t)
                tokens += n
            self._pace(len(batch), tokens)
            try:
                out.extend(self._call(batch, task))
                i += len(batch)
                strikes = 0
            except RateLimited as e:
                self.rate_limited += 1
                strikes += 1
                if strikes > self.max_consecutive_429:
                    raise RuntimeError(f"Gemini keeps rate-limiting ({e.quota}); stopping") from e
                if len(batch) > 1:
                    self.max_batch = max(1, len(batch) // 2)
                    self.max_batch_tokens = max(1_000, tokens // 2)
                log.warning("Gemini 429 %s: waiting %.0fs, batch now <= %d texts / ~%d tokens",
                            e.quota, e.wait, self.max_batch, self.max_batch_tokens)
                time.sleep(e.wait)
        return out


class LocalEmbedder:
    def __init__(self, model: str, dim: int):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError('Local embeddings need: pip install -e ".[local]"') from e
        self.model, self.dim = model, dim
        self.impl = TextEmbedding(model_name=model)
        self.tokens_sent = 0
        self.requests = 0

    def embed(self, texts: list[str], task: str) -> list[np.ndarray]:
        it = self.impl.query_embed(texts) if task == "RETRIEVAL_QUERY" else self.impl.embed(texts)
        vecs = [_normalise(np.asarray(v, dtype=np.float32)) for v in it]
        if vecs and len(vecs[0]) != self.dim:
            raise RuntimeError(f"{self.model} gives {len(vecs[0])} dims, EMBED_DIM is {self.dim}")
        return vecs


class Embedder:
    """Cached front end: embed_documents / embed_query."""

    def __init__(self, provider: str, model: str, dim: int, api_key: str, cache_dir: Path,
                 rpm: int = 0, tpm: int = 0):
        if provider == "gemini":
            self.impl = GeminiEmbedder(api_key, model, dim, rpm, tpm)
        elif provider == "local":
            self.impl = LocalEmbedder(model, dim)
        else:
            raise ValueError(f"Unknown EMBED_PROVIDER {provider!r}")
        self.model, self.dim = model, dim
        self.cache = EmbeddingCache(cache_dir / "embeddings.sqlite")
        self.cache_hits = 0

    def _embed(self, texts: list[str], task: str) -> list[np.ndarray]:
        keys = [EmbeddingCache.key(self.model, self.dim, task, t) for t in texts]
        found = self.cache.get_many(keys)
        self.cache_hits += len(found)
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in found]
        if todo:
            vecs = self.impl.embed([t for _, t in todo], task)
            new = dict(zip([k for k, _ in todo], vecs))
            self.cache.put_many(new)
            found.update(new)
        return [found[k] for k in keys]

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def make_embedder(settings) -> Embedder:
    return Embedder(
        settings.embed_provider,
        settings.embed_model,
        settings.embed_dim,
        settings.gemini_api_key,
        settings.cache_dir,
        settings.embed_rpm,
        settings.embed_tpm,
    )
