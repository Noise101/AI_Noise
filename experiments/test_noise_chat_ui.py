import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

import noise_chat_ui as ui


class NoiseChatUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.temp.name)
        self.server = ui.create_server(self.runtime, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        encoded = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, response.headers, payload

    def test_index_and_state_are_available(self):
        status, headers, payload = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Noiseとの対話", payload.decode())
        self.assertIn("Content-Security-Policy", headers)
        status, _, payload = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["conversation"]["turns"], 0)

    def test_chat_persists_grounded_memory(self):
        status, _, payload = self.request("POST", "/api/chat", {"message": "レモンは果物だよ"})
        self.assertEqual(status, 200)
        result = json.loads(payload)
        self.assertIn("まだ私自身では確かめていません", result["reply"])
        persisted = ui.read_json(self.runtime / ui.CONVERSATION_FILE)
        self.assertEqual(persisted["claims"]["レモン"][0]["predicate"], "果物")

    def test_invalid_requests_are_rejected(self):
        status, _, _ = self.request("POST", "/api/chat", {"message": ""})
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/../noise_chat_v1.py")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
