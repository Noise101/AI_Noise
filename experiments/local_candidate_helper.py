"""Optional local-LLM helper. Its outputs are proposals, never evidence."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class LocalProposal:
    model: str
    task: str
    candidates: list[dict]
    verified: bool = False
    evidence_score: float = 0.0


class OllamaCandidateHelper:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3:4b")

    def available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=2) as response:
                models = json.load(response).get("models", [])
            return any(item.get("name") == self.model for item in models)
        except Exception:
            return False

    def propose_japanese_senses(self, form: str, context: str, limit: int = 4) -> LocalProposal | None:
        if not self.available():
            return None
        prompt = (
            "次の日本語表層形について、この文脈で可能な意味候補を最大"
            f"{limit}件だけJSONで提案してください。説明や結論は禁止。"
            "各候補は label, observable_features, search_query を持つ。"
            "labelは表層形を繰り返さず、漢字表記または意味カテゴリで候補を区別する。"
            "これは未検証候補であり、文脈にない事実を追加しない。\n"
            f"表層形: {form}\n文脈: {context[:1600]}"
        )
        schema = {
            "type": "object", "properties": {"candidates": {"type": "array", "maxItems": limit,
                "items": {"type": "object", "properties": {
                    "label": {"type": "string"}, "observable_features": {"type": "array", "items": {"type": "string"}},
                    "search_query": {"type": "string"}},
                    "required": ["label", "observable_features", "search_query"]}}},
            "required": ["candidates"],
        }
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                              "think": False, "format": schema, "options": {"temperature": 0}}).encode()
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.load(response)
            parsed = json.loads(result.get("response", "{}"))
            candidates = []
            seen_labels = set()
            for candidate in parsed.get("candidates", [])[:limit]:
                if not all(key in candidate for key in ("label", "observable_features", "search_query")):
                    continue
                label = str(candidate["label"]).strip()[:80]
                if not label or label == form or label in seen_labels:
                    continue
                seen_labels.add(label)
                candidates.append({
                    "label": label,
                    "observable_features": [str(item)[:120] for item in candidate["observable_features"][:8]],
                    "search_query": str(candidate["search_query"])[:160],
                })
            return LocalProposal(self.model, "propose_japanese_senses", candidates)
        except Exception:
            return None


class NullCandidateHelper:
    def available(self) -> bool:
        return False

    def propose_japanese_senses(self, form: str, context: str, limit: int = 4) -> None:
        return None
