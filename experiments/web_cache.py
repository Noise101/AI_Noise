"""Small persistent cache for reproducible, low-request read-only web experiments."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path


class NetworkBudgetExceeded(RuntimeError):
    pass


class ReadOnlyWebCache:
    def __init__(self, cache_dir: Path | None = None, ttl_seconds: int = 7 * 24 * 3600):
        default = Path(__file__).resolve().parent.parent / ".cache" / "web"
        self.cache_dir = cache_dir or Path(os.environ.get("AI_NOISE_CACHE_DIR", default))
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.network_requests = 0
        self.network_limit: int | None = None
        self.budget_start = 0

    def set_network_budget(self, additional_requests: int | None) -> None:
        self.network_limit = additional_requests
        self.budget_start = self.network_requests

    def remaining_network_budget(self) -> int | None:
        if self.network_limit is None:
            return None
        return max(0, self.network_limit - (self.network_requests - self.budget_start))

    def get_bytes(self, url: str, user_agent: str, accept: str = "*/*") -> bytes:
        key = hashlib.sha256(url.encode()).hexdigest()
        data_path = self.cache_dir / f"{key}.bin"
        meta_path = self.cache_dir / f"{key}.json"
        cache_enabled = os.environ.get("AI_NOISE_DISABLE_CACHE") != "1"
        if cache_enabled and data_path.exists() and meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if time.time() - metadata["fetched_at"] <= self.ttl_seconds:
                    self.hits += 1
                    return data_path.read_bytes()
            except (OSError, ValueError, KeyError):
                pass
        self.misses += 1
        request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": accept})
        payload = None
        last_error = None
        for attempt in range(3):
            if self.network_limit is not None and self.network_requests - self.budget_start >= self.network_limit:
                raise NetworkBudgetExceeded(f"network request budget exhausted before {url}")
            self.network_requests += 1
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    payload = response.read()
                break
            except urllib.error.HTTPError:
                # 4xx/5xx policy belongs to the caller; blindly repeating a permanent 403 is harmful.
                raise
            except (http.client.IncompleteRead, TimeoutError, ConnectionResetError,
                    urllib.error.URLError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.25 * (2 ** attempt))
        if payload is None:
            raise last_error or RuntimeError(f"empty network response for {url}")
        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            data_path.write_bytes(payload)
            meta_path.write_text(json.dumps({"url": url, "fetched_at": time.time()}) + "\n", encoding="utf-8")
        return payload

    def get_json(self, url: str, user_agent: str) -> dict:
        return json.loads(self.get_bytes(url, user_agent, "application/json"))

    def stats(self) -> dict:
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "network_requests": self.network_requests,
                "remaining_network_budget": self.remaining_network_budget()}


WEB_CACHE = ReadOnlyWebCache()
