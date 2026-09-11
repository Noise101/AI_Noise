#!/usr/bin/env python3
"""Local browser UI for direct, grounded conversation with Noise."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import noise_chat_v1 as noise_chat


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_RUNTIME = REPO / ".local"
STATIC = HERE / "noise_chat_ui"
CONVERSATION_FILE = "human-conversation.json"
WORD_MEMORY_FILE = "reading-word-meaning.json"
STATUS_FILE = "status.json"
UI_STATE_FILE = "chat-ui-status.json"
MAX_BODY = 8 * 1024
LOCK = threading.Lock()

PHASES_JA = {
    "learning": "学習中",
    "japanese_only": "日本語を学習中",
    "normal_curriculum": "通常教材を学習中",
    "counterexample_hunt": "反例を探索中",
    "between_rounds": "次の処理を準備中",
    "supervisor_retry_wait": "再試行待ち",
    "worker_error_wait": "エラー後の再試行待ち",
    "japanese_only_error": "日本語学習でエラー",
    "stopped_by_user": "停止済み",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def process_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def build_state(runtime: Path) -> dict:
    conversation = read_json(runtime / CONVERSATION_FILE)
    worker = read_json(runtime / STATUS_FILE)
    pid = worker.get("pid")
    chat_summary = noise_chat.summary(conversation)
    return {
        "conversation": {
            **chat_summary,
            "turns_recent": (conversation.get("turns") or [])[-80:],
            "unknown_topics_recent": list((conversation.get("unknown_topics") or {}).keys())[-8:],
        },
        "worker": {
            "alive": process_alive(pid),
            "pid": pid,
            "phase": worker.get("phase", "unknown"),
            "phase_ja": PHASES_JA.get(worker.get("phase"), worker.get("phase", "不明")),
            "seed": worker.get("seed"),
            "heartbeat": worker.get("heartbeat"),
            "error": worker.get("error"),
        },
        "principle": "会話は経験として記憶します。人の発言を未検証の世界事実にはしません。",
    }


class NoiseChatHandler(BaseHTTPRequestHandler):
    server_version = "NoiseChat/1"

    @property
    def runtime(self) -> Path:
        return self.server.runtime  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'")
        self.end_headers()

    def _send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(payload))
        self.wfile.write(payload)

    def _send_static(self, filename: str, content_type: str) -> None:
        path = STATIC / filename
        try:
            payload = path.read_bytes()
        except OSError:
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        self._headers(HTTPStatus.OK, content_type, len(payload))
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            self._send_static("index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._send_static("styles.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._send_static("app.js", "text/javascript; charset=utf-8")
        elif path == "/api/state":
            self._send_json(build_state(self.runtime))
        elif path == "/api/health":
            self._send_json({"ok": True, "service": "noise-chat"})
        else:
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urllib.parse.urlsplit(self.path).path != "/api/chat":
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._send_json({"error": "invalid_body_size"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, str) or not message.strip() or len(message) > 500:
            self._send_json({"error": "message_must_be_1_to_500_characters"}, HTTPStatus.BAD_REQUEST)
            return
        with LOCK:
            memory_path = self.runtime / CONVERSATION_FILE
            reply, memory = noise_chat.converse(
                message, read_json(memory_path), read_json(self.runtime / WORD_MEMORY_FILE))
            write_json(memory_path, memory)
        self._send_json({"reply": reply, "state": build_state(self.runtime)})


class NoiseChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: Path):
        self.runtime = runtime
        super().__init__(address, NoiseChatHandler)


def create_server(runtime: Path, host: str = "127.0.0.1", port: int = 8765) -> NoiseChatServer:
    return NoiseChatServer((host, port), runtime.resolve())


def serve(runtime: Path, host: str, port: int) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    server = create_server(runtime, host, port)
    actual_port = server.server_address[1]
    write_json(runtime / UI_STATE_FILE, {
        "pid": os.getpid(), "host": host, "port": actual_port,
        "url": f"http://{host}:{actual_port}/", "started_at": time.time(),
    })
    print(f"Noise GUI: http://{host}:{actual_port}/", flush=True)
    server.serve_forever(poll_interval=0.25)


def start(runtime: Path, host: str, port: int) -> None:
    prior = read_json(runtime / UI_STATE_FILE)
    if process_alive(prior.get("pid")):
        print(json.dumps({"status": "already_running", **prior}, ensure_ascii=False))
        return
    runtime.mkdir(parents=True, exist_ok=True)
    log = open(runtime / "chat-ui.log", "a", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "serve",
               "--runtime", str(runtime), "--host", host, "--port", str(port)]
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    log.close()
    for _ in range(40):
        time.sleep(0.05)
        state = read_json(runtime / UI_STATE_FILE)
        if state.get("pid") == process.pid:
            print(json.dumps({"status": "started", **state}, ensure_ascii=False))
            return
        if process.poll() is not None:
            break
    raise RuntimeError(f"GUI did not start; inspect {runtime / 'chat-ui.log'}")


def stop(runtime: Path) -> None:
    state = read_json(runtime / UI_STATE_FILE)
    pid = state.get("pid")
    if not process_alive(pid):
        print(json.dumps({"status": "not_running"}, ensure_ascii=False))
        return
    os.kill(pid, signal.SIGTERM)
    print(json.dumps({"status": "stop_requested", "pid": pid}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("serve", "start"):
        child = subparsers.add_parser(command)
        child.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
        child.add_argument("--host", default="127.0.0.1")
        child.add_argument("--port", type=int, default=8765)
    for command in ("status", "stop"):
        child = subparsers.add_parser(command)
        child.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.runtime, args.host, args.port)
    elif args.command == "start":
        start(args.runtime, args.host, args.port)
    elif args.command == "status":
        state = read_json(args.runtime / UI_STATE_FILE)
        state["alive"] = process_alive(state.get("pid"))
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        stop(args.runtime)


if __name__ == "__main__":
    main()
