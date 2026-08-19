"""Central configuration. Everything tunable lives here."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class Config:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    chat_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
    )

    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 150))
    min_chunk_chars: int = 120

    top_k: int = int(os.getenv("TOP_K", 5))
    search_type: str = os.getenv("SEARCH_TYPE", "mmr")
    mmr_fetch_k: int = int(os.getenv("MMR_FETCH_K", 20))

    temperature: float = float(os.getenv("TEMPERATURE", 0.1))

    persist_dir: Path = ROOT / "data" / "chroma"
    upload_dir: Path = ROOT / "data" / "papers"

    drop_references: bool = os.getenv("DROP_REFERENCES", "true").lower() == "true"

    def validate(self) -> None:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to the .env file."
            )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
