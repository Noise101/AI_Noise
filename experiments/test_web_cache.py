import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
