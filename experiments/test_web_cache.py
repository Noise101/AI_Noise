import json
import http.client
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from web_cache import NetworkBudgetExceeded, ReadOnlyWebCache


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class WebCacheTest(unittest.TestCase):
    def test_second_identical_read_uses_cache_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ReadOnlyWebCache(Path(directory), ttl_seconds=60)
            payload = json.dumps({"value": 1}).encode()
            with patch("urllib.request.urlopen", return_value=FakeResponse(payload)) as mocked:
                self.assertEqual(cache.get_json("https://example.test/data", "test"), {"value": 1})
                self.assertEqual(cache.get_json("https://example.test/data", "test"), {"value": 1})
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(cache.stats()["network_requests"], 1)
            self.assertEqual(cache.stats()["cache_hits"], 1)

    def test_network_budget_is_enforced_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ReadOnlyWebCache(Path(directory))
            cache.set_network_budget(0)
            with patch("urllib.request.urlopen") as mocked:
                with self.assertRaises(NetworkBudgetExceeded):
                    cache.get_bytes("https://example.test/new", "test")
            mocked.assert_not_called()

    def test_incomplete_download_is_retried_and_only_complete_payload_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ReadOnlyWebCache(Path(directory))
            complete = FakeResponse(b"complete")
            with patch("urllib.request.urlopen", side_effect=[
                    http.client.IncompleteRead(b"partial", 3), complete]) as mocked:
                self.assertEqual(cache.get_bytes("https://example.test/book", "test"), b"complete")
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(cache.stats()["network_requests"], 2)

    def test_permanent_http_denial_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ReadOnlyWebCache(Path(directory))
            denial = urllib.error.HTTPError("https://example.test/denied", 403,
                                            "Forbidden", {}, None)
            with patch("urllib.request.urlopen", side_effect=denial) as mocked:
                with self.assertRaises(urllib.error.HTTPError):
                    cache.get_bytes("https://example.test/denied", "test")
            self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
