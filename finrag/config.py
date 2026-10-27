"""Settings from environment (.env) and the company list in config/companies.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    cik: int
    financebench: str
    fiscal_years: tuple[int, ...]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    sec_user_agent: str
    gemini_api_key: str
    embed_provider: str
    embed_model: str
    embed_dim: int
    embed_rpm: int = 0  # texts per minute, 0 = no pacing
    embed_tpm: int = 0  # estimated tokens per minute, 0 = no pacing

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw" / "edgar"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    provider = os.getenv("EMBED_PROVIDER", "gemini").strip().lower()
    default_model = "gemini-embedding-001" if provider == "gemini" else "BAAI/bge-base-en-v1.5"
    return Settings(
        data_dir=data_dir,
        database_url=os.getenv("DATABASE_URL", "postgresql://finrag:finrag@localhost:5433/finrag"),
        sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        embed_provider=provider,
        embed_model=os.getenv("EMBED_MODEL", "").strip() or default_model,
        embed_dim=int(os.getenv("EMBED_DIM", "768")),
        embed_rpm=int(os.getenv("EMBED_RPM") or 0),
        embed_tpm=int(os.getenv("EMBED_TPM") or 0),
    )


def load_companies(path: Path | None = None) -> list[Company]:
    path = path or ROOT / "config" / "companies.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return [
        Company(
            name=c["name"],
            ticker=c["ticker"],
            cik=int(c["cik"]),
            financebench=c.get("financebench", c["name"]),
            fiscal_years=tuple(c["fiscal_years"]),
        )
        for c in data["company"]
    ]
