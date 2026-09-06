#!/usr/bin/env python3
"""Local-LLM reading scaffold: rephrase a story Noise cannot parse into simple
SOV sentences it can.

A caregiver reading with a child paraphrases the hard passages ("つまり、きつねは
ぶどうが食べたかったんだよ").  This does the same with the local model: it asks
for short "だれが / なにを / どうした" sentences, then **verifies every output**
before use --

  * the rephrasing must parse into >= MIN_EVENTS events (the whole point)
  * it must keep the original's content: a large fraction of the source's
    kanji / katakana content tokens must survive
  * it must not be substantially longer than the source

The model is never an authority.  Its output is a *reading aid*: scaffolded
events let a book be understood and graduate instead of getting stuck, but they
are kept out of the frozen comprehension benchmark and the character RNN, which
stay on text Noise actually read.  Nothing here updates a belief (constitution:
local model at evidence score zero).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

import japanese_event_v1 as jevent

MIN_EVENTS = 3
MAX_LENGTH_RATIO = 1.4
MIN_CONTENT_KEPT = 0.35
MAX_SENTENCES_IN = 30

_JP = re.compile(r"[぀-ヿ㐀-鿿]")
_JP_RUN = re.compile(r"[぀-ヿ㐀-鿿]+")
_KANJI_KATA = re.compile(r"[㐀-鿿]{2,}|[゠-ヿ]{2,}")     # kanji / katakana runs, len >= 2
_NOUN_PARTICLE = re.compile(r"([぀-ヿ㐀-鿿]{2,})(?:が|を|に|は|も|へ|と|で)$")
_SENT = re.compile(r"(?<=[。！？])")

_PROMPT = """次の日本語の物語を、小学1年生にもわかるように、みじかい文にいいかえてください。
まもってほしいルール:
- 1文につき「だれが」「なにを」「どうした」を1つずつ。
- 文の中のことばを、半角スペースで区切る（れい: こうもりが 地べたに おちました）。
- 「」の中の会話は、地の文に書きなおす（れい:「たすけて」と言った → こうもりは たすけてと たのみました）。
- 話のじゅんばんと内ようは、かえない。むずかしい漢字は、やさしいことばにする。
物語:
{story}"""


class OllamaReader:
    """Minimal client for a local Ollama model, JSON-schema constrained output."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3.8:27b")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as response:
                models = json.load(response).get("models", [])
            return any(item.get("name") == self.model for item in models)
        except Exception:
            return False

    def simplify(self, story: str) -> list[str]:
        schema = {"type": "object",
                  "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
                  "required": ["sentences"]}
        payload = json.dumps({
            "model": self.model, "prompt": _PROMPT.format(story=story[:2000]),
            "stream": False, "think": False, "format": schema,
            "options": {"temperature": 0.2, "num_predict": 600}}).encode()
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                parsed = json.loads(json.load(response).get("response", "{}"))
        except Exception:
            return []
        return [str(s).strip() for s in parsed.get("sentences", []) if str(s).strip()][:40]


def _recurring_terms(text: str) -> set[str]:
    """The story's content words, approximated without a tokeniser: 2-4 char
    Japanese n-grams that recur (its entities and key verbs), plus every
    multi-char kanji / katakana run."""
    terms = set(_KANJI_KATA.findall(text))
    counts: dict[str, int] = {}
    for run in _JP_RUN.findall(text):
        for n in (2, 3, 4):
            for i in range(len(run) - n + 1):
                gram = run[i:i + n]
                counts[gram] = counts.get(gram, 0) + 1
    terms.update(g for g, c in counts.items() if c >= 2 and len(g) >= 2)
    if len(terms) < 4:                       # short story: fall back to kanji bigrams
        terms.update(g for g in counts if len(g) == 2 and re.search(r"[㐀-鿿]", g))
    return terms


def _content_kept(source: str, simple: str) -> float:
    terms = _recurring_terms(source)
    if not terms:
        return 1.0
    return sum(t in simple for t in terms) / len(terms)


def _harvest_known_words(sentences: list[str]) -> set[str]:
    """The model space-separates bunsetsu, so the spacing marks word boundaries."""
    known: set[str] = set()
    for sentence in sentences:
        for token in sentence.split():
            token = token.strip("、。「」")
            match = _NOUN_PARTICLE.match(token)
            if match:
                known.add(match.group(1))
            elif re.fullmatch(r"[぀-ヿ㐀-鿿]{2,6}", token):
                known.add(token)
    return known


def simplify_story(text: str, worker: OllamaReader | None = None,
                   base_known_words: set[str] | None = None) -> dict:
    """Return a verified simple-Japanese rephrasing of `text`, or a rejection."""
    worker = worker or OllamaReader()
    source_sentences = [s for s in _SENT.split(text) if len(_JP.findall(s)) >= 3]
    if len(source_sentences) > MAX_SENTENCES_IN:
        return {"status": "skipped", "reason": "story too long for a single pass"}

    simple_sentences = worker.simplify(text)
    if not simple_sentences:
        return {"status": "no_response"}

    known = set(base_known_words or set()) | _harvest_known_words(simple_sentences)
    simple_text = "".join(s.replace(" ", "").strip() for s in simple_sentences)
    events = [e.__dict__ for e in jevent.extract_story(simple_text, known)]

    kept = _content_kept(text, simple_text)
    length_ratio = len(simple_text) / max(1, len(text.replace("\n", "")))

    checks = {
        "parses_to_events": len(events) >= MIN_EVENTS,
        "keeps_content": kept >= MIN_CONTENT_KEPT,
        "not_much_longer": length_ratio <= MAX_LENGTH_RATIO,
    }
    verified = all(checks.values())
    return {
        "status": "simplified" if verified else "rejected",
        "verified": verified,
        "checks": checks,
        "content_kept": round(kept, 3),
        "length_ratio": round(length_ratio, 3),
        "event_count": len(events),
        "events": events if verified else [],
        "text": simple_text if verified else "",
        "sentences": simple_sentences,
        "known_words_added": sorted(_harvest_known_words(simple_sentences)),
        "note": "local-model reading scaffold, verified to parse and preserve "
                "content; not evidence, kept out of the frozen benchmark and RNN",
    }
