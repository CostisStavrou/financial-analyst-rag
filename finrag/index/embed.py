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


class GeminiEmbedder:
    max_batch = 100  # texts per request
    max_batch_tokens = 20_000  # keeps single requests well inside per-minute token quotas

    def __init__(self, api_key: str, model: str, dim: int):
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

    def _call(self, texts: list[str], task: str) -> list[np.ndarray]:
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim)
        for attempt in range(8):
            try:
                self.requests += 1
                res = self.client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                self.tokens_sent += sum(est_tokens(t) for t in texts)
                return [_normalise(np.asarray(e.values, dtype=np.float32)) for e in res.embeddings]
            except errors.APIError as e:
                if e.code not in (429, 500, 503) or attempt == 7:
                    raise
                hint = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE)
                wait = float(hint.group(1)) + 1 if hint else min(60, 2 ** (attempt + 1))
                log.warning("Gemini %s; waiting %.0fs", e.code, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def embed(self, texts: list[str], task: str) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        batch: list[str] = []
        for t in texts:
            if batch and (
                len(batch) >= self.max_batch
                or sum(map(est_tokens, batch)) + est_tokens(t) > self.max_batch_tokens
            ):
                out.extend(self._call(batch, task))
                batch = []
            batch.append(t)
        if batch:
            out.extend(self._call(batch, task))
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

    def __init__(self, provider: str, model: str, dim: int, api_key: str, cache_dir: Path):
        if provider == "gemini":
            self.impl = GeminiEmbedder(api_key, model, dim)
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
    )
