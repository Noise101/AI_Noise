"""Structural Kanjipedia lookup without copying copyrighted dictionary prose."""

from __future__ import annotations

import hashlib
import html
import re
import urllib.parse

from web_cache import WEB_CACHE


USER_AGENT = "AI_Noise/0.22 (read-only exact-entry validation; https://github.com/Noise101/AI_Noise)"
ENTRY = re.compile(r'<a\s+href="(?P<path>/(?:kanji|kotoba)/[^"?#]+)"[^>]*>\s*(?P<label>[^<]+?)\s*</a>')


def exact_entry_from_html(form: str, page: str) -> dict | None:
    for match in ENTRY.finditer(page):
        label = html.unescape(match.group("label")).strip()
        if label == form:
            return {"path": match.group("path"), "label": label}
    return None


class KanjipediaReference:
    name = "Kanjipedia"

    def lookup(self, form: str) -> dict | None:
        target = "kt" if len(form) == 1 else "wt"
        query = urllib.parse.urlencode({"k": form, target: 1, "sk": "perfect"})
        search_url = "https://www.kanjipedia.jp/search?" + query
        try:
            body = WEB_CACHE.get_bytes(search_url, USER_AGENT)
        except Exception:
            return None
        page = body.decode("utf-8", errors="replace")
        entry = exact_entry_from_html(form, page)
        if not entry:
            return None
        return {
            "source": self.name,
            "url": "https://www.kanjipedia.jp" + entry["path"],
            "requested_title": form,
            "matched_title": entry["label"],
            "exact_match": True,
            "sha256": hashlib.sha256(body).hexdigest(),
            "evidence_scope": "entry existence only; definition prose is not copied",
        }
