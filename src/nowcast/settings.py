"""Runtime settings loaded from the environment / .env file."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PLACEHOLDERS = {"you@example.com"}


class SettingsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    contact_email: str | None
    db_path: Path
    cache_dir: Path
    universe_path: Path
    offline: bool

    def require_email(self) -> str:
        """Return the contact email, refusing missing or placeholder values (SEC/Wikimedia policy)."""
        e = (self.contact_email or "").strip()
        if not e or e in _PLACEHOLDERS or not _EMAIL_RE.match(e):
            raise SettingsError(
                "CONTACT_EMAIL is missing or a placeholder. Copy .env.example to .env and set it; "
                "SEC and Wikimedia require a contact address in the User-Agent."
            )
        return e


def load_settings(env_file: Path | None = None) -> Settings:
    file_vals = dotenv_values(env_file or PROJECT_ROOT / ".env")

    def get(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key) or file_vals.get(key) or default

    return Settings(
        contact_email=get("CONTACT_EMAIL"),
        db_path=Path(get("NOWCAST_DB", str(PROJECT_ROOT / "data" / "nowcast.duckdb"))),
        cache_dir=Path(get("NOWCAST_CACHE", str(PROJECT_ROOT / ".cache" / "raw"))),
        universe_path=Path(get("NOWCAST_UNIVERSE", str(PROJECT_ROOT / "config" / "universe.yaml"))),
        offline=(get("NOWCAST_OFFLINE", "0") or "0").lower() in {"1", "true", "yes"},
    )
