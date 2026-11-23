"""Polite, cached HTTP client: declared User-Agent, rate limit, on-disk cache of raw responses."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import httpx

from nowcast.settings import Settings

log = logging.getLogger(__name__)

SEC_MAX_PER_SEC = 10  # SEC fair-access policy hard limit
RETRY_STATUS = {429, 500, 502, 503, 504}


class OfflineCacheMiss(RuntimeError):
    pass


def make_user_agent(email: str) -> str:
    return f"alt-data-kpi-nowcast/0.1 (research; {email})"


class RateLimiter:
    """Minimum spacing between calls; with max_per_sec <= 10 no 1-second window exceeds 10 calls."""

    def __init__(
        self,
        max_per_sec: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not 0 < max_per_sec <= SEC_MAX_PER_SEC:
            raise ValueError(f"max_per_sec must be in (0, {SEC_MAX_PER_SEC}]")
        self.interval = 1.0 / max_per_sec
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self._last + self.interval - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
        self._last = now


class CachedClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_per_sec: float,
        offline: bool = False,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_tries: int = 3,
    ):
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.limiter = RateLimiter(max_per_sec, clock=clock, sleep=sleep)
        self._sleep = sleep
        self.max_tries = max_tries
        self.requests_made = 0
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60.0,
            transport=transport,
            follow_redirects=True,
        )

    def get_json(self, url: str, cache_key: str) -> dict:
        path = self.cache_dir / cache_key
        if path.exists():
            return json.loads(path.read_text())
        if self.offline:
            raise OfflineCacheMiss(f"offline and not cached: {cache_key}")
        for attempt in range(1, self.max_tries + 1):
            self.limiter.wait()
            self.requests_made += 1
            resp = self._http.get(url)
            if resp.status_code in RETRY_STATUS and attempt < self.max_tries:
                backoff = 2.0 ** attempt
                log.warning("HTTP %s for %s; retry %d in %.0fs", resp.status_code, url, attempt, backoff)
                self._sleep(backoff)
                continue
            resp.raise_for_status()
            data = resp.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)  # atomic: no half-written cache files
            return data
        raise AssertionError("unreachable")  # pragma: no cover

    def get_text(self, url: str, cache_key: str) -> str:
        """Like get_json, for HTML/text documents (EDGAR archives)."""
        path = self.cache_dir / cache_key
        if path.exists():
            return path.read_text(errors="ignore")
        if self.offline:
            raise OfflineCacheMiss(f"offline and not cached: {cache_key}")
        for attempt in range(1, self.max_tries + 1):
            self.limiter.wait()
            self.requests_made += 1
            resp = self._http.get(url)
            if resp.status_code in RETRY_STATUS and attempt < self.max_tries:
                self._sleep(2.0 ** attempt)
                continue
            resp.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(resp.text)
            tmp.replace(path)
            return resp.text
        raise AssertionError("unreachable")  # pragma: no cover

    def close(self) -> None:
        self._http.close()


def sec_client(settings: Settings) -> CachedClient:
    email = settings.require_email() if not settings.offline else (settings.contact_email or "offline")
    return CachedClient(make_user_agent(email), settings.cache_dir, max_per_sec=8, offline=settings.offline)


def wiki_client(settings: Settings) -> CachedClient:
    email = settings.require_email() if not settings.offline else (settings.contact_email or "offline")
    return CachedClient(make_user_agent(email), settings.cache_dir, max_per_sec=5, offline=settings.offline)
