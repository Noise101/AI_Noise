#!/usr/bin/env python3
"""Fetch Japanese children's stories for the developmental reading curriculum.

Read-only, cached, budget-aware (same ReadOnlyWebCache the English path uses),
and paced with a bounded 429 backoff so a batch of fetches does not trip the
Wikimedia / Aozora rate limit.

Sources: ja.wikisource.org 「イソップ童話集」 (~37 short all-kana Aesop fables, the
best beginner material there) plus 猿蟹合戦; Aozora Bunko for the 楠山正雄 folktale
retellings and other children's authors (AOZORA_AUTHORS).  Text comes back as
plain UTF-8; ruby and editorial headers are stripped.

`kernel_titles()` is the bootstrap list (seed + live Aesop subpages), needed to
start from ~zero known Japanese vocabulary before autonomous discovery takes over.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass

from web_cache import WEB_CACHE, NetworkBudgetExceeded

API = "https://ja.wikisource.org/w/api.php"
UA = "AI_Noise/0.30 (developmental Japanese reading; read-only)"

# Pacing so a batch of fetches does not trip the Wikimedia / Aozora rate limit.
_FETCH_PACING_SECONDS = 0.6
_RATE_LIMIT_BACKOFF = (3.0, 8.0)
_last_fetch_at = 0.0

# ja.wikisource has almost no folktales, but 「イソップ童話集」 has ~37 short
# all-kana subpages (pulled live by aesop_kernel) and 猿蟹合戦 is the one folktale
# page that reliably exists.  The 楠山正雄 folktale retellings live on Aozora and
# come in through AOZORA_AUTHORS.
KERNEL_SEED = ("猿蟹合戦",)
AESOP_PREFIX = "イソップ童話集/"


def _pace() -> None:
    global _last_fetch_at
    wait = _FETCH_PACING_SECONDS - (time.monotonic() - _last_fetch_at)
    if wait > 0:
        time.sleep(wait)
    _last_fetch_at = time.monotonic()


def _get_json(url: str) -> dict:
    """WEB_CACHE.get_json with pacing and a bounded 429 backoff."""
    for delay in (0.0,) + _RATE_LIMIT_BACKOFF:
        if delay:
            time.sleep(delay)
        _pace()
        try:
            return WEB_CACHE.get_json(url, UA)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
    return {}


def _get_bytes(url: str, user_agent: str, accept: str) -> bytes:
    for delay in (0.0,) + _RATE_LIMIT_BACKOFF:
        if delay:
            time.sleep(delay)
        _pace()
        try:
            return WEB_CACHE.get_bytes(url, user_agent, accept)
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
    raise urllib.error.HTTPError(url, 429, "rate limited after backoff", {}, None)


def aesop_kernel(limit: int = 40) -> tuple[str, ...]:
    """Live list of 「イソップ童話集」 subpages -- short all-kana Aesop fables, the
    best beginner material actually on ja.wikisource."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "allpages", "apprefix": AESOP_PREFIX,
        "apnamespace": 0, "aplimit": limit, "format": "json", "formatversion": 2})
    try:
        data = _get_json(f"{API}?{params}")
    except (NetworkBudgetExceeded, Exception):
        return ()
    return tuple(p["title"] for p in data.get("query", {}).get("allpages", []))


def kernel_titles(limit: int = 40) -> tuple[str, ...]:
    return KERNEL_SEED + aesop_kernel(limit)


# Back-compat: callers that still read corpus.KERNEL directly get the seed only.
KERNEL = KERNEL_SEED

_HEADING = re.compile(r"^={1,6}[^=]+={1,6}$", re.M)
_RUBY = re.compile(r"《[^》]*》|[｜|]")
_NOTES = re.compile(r"(底本[:：].*|この作品は.*|パブリックドメイン.*|翻訳者[:：].*)", re.S)
_BRACKET = re.compile(r"[\[［].{0,40}?[\]］]")
JP_CHARS = re.compile(r"[぀-ヿ㐀-鿿]")


@dataclass
class JapaneseText:
    title: str
    url: str
    text: str

    @property
    def japanese_char_count(self) -> int:
        return len(JP_CHARS.findall(self.text))


def _clean(raw: str) -> str:
    text = _NOTES.sub("", raw)
    text = _HEADING.sub("", text)
    text = _RUBY.sub("", text)
    text = _BRACKET.sub("", text)
    text = re.sub(r"[ \t　]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def fetch(title: str) -> JapaneseText | None:
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": title, "format": "json", "formatversion": 2})
    try:
        data = _get_json(f"{API}?{params}")
    except (NetworkBudgetExceeded, Exception):
        return None
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    page = pages[0]
    text = _clean(page.get("extract", ""))
    if len(JP_CHARS.findall(text)) < 120:
        return None
    url = "https://ja.wikisource.org/wiki/" + urllib.parse.quote(
        page["title"].replace(" ", "_"), safe="/()")
    return JapaneseText(title=page["title"], url=url, text=text)


def search(query: str, limit: int = 5) -> list[str]:
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query, "srnamespace": 0,
        "srlimit": limit, "format": "json", "formatversion": 2})
    try:
        data = _get_json(f"{API}?{params}")
    except (NetworkBudgetExceeded, Exception):
        return []
    return [item["title"] for item in data.get("query", {}).get("search", [])]


AOZORA_UA = "AI_Noise/0.30 (developmental Japanese reading; read-only)"
# main_text contains nested divs; take everything up to the bibliographic block
_MAIN_TEXT = re.compile(
    r'<div class="main_text">(.*?)(?:<div class="bibliographical_information">|</body>)',
    re.S)
_RT = re.compile(r"<rp>.*?</rp>|<rt>.*?</rt>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _aozora_title(html: str) -> str:
    for pat in (r'<h1[^>]*class="title"[^>]*>(.*?)</h1>',
                r'<meta\s+name="DC\.Title"\s+content="([^"]+)"'):
        m = re.search(pat, html, re.S)
        if m:
            return _TAG.sub("", m.group(1)).strip()
    return ""


def fetch_aozora(html_url: str) -> JapaneseText | None:
    """A work's XHTML file on aozora.gr.jp (Shift_JIS), ruby readings dropped."""
    try:
        raw = _get_bytes(html_url, AOZORA_UA, "text/html")
    except (NetworkBudgetExceeded, Exception):
        return None
    html = raw.decode("shift_jis", errors="replace")
    body = _MAIN_TEXT.search(html)
    if not body:
        return None
    text = _RT.sub("", body.group(1))
    text = _TAG.sub("", text).replace("｜", "")
    text = re.sub(r"[ \t　]+", "", re.sub(r"\n{2,}", "\n", text)).strip()
    if len(JP_CHARS.findall(text)) < 120:
        return None
    title = _aozora_title(html)
    return JapaneseText(title=title or html_url.rsplit("/", 1)[-1], url=html_url, text=text)


def category_members(category: str, limit: int = 100) -> list[str]:
    """Titles in a ja.wikisource category (e.g. 'Category:童話')."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "categorymembers", "cmtitle": category,
        "cmnamespace": 0, "cmlimit": limit, "cmtype": "page",
        "format": "json", "formatversion": 2})
    try:
        data = _get_json(f"{API}?{params}")
    except (NetworkBudgetExceeded, Exception):
        return []
    return [item["title"] for item in data.get("query", {}).get("categorymembers", [])]


# Aozora author "person" pages -> their public-domain works (children's authors
# and translators of children's classics).  Used to widen the shelf once the
# kernel is exhausted, cheapest first.
AOZORA_AUTHORS = {
    "楠山正雄": 329, "新美南吉": 121, "小川未明": 1475, "宮沢賢治": 81,
    "浜田広介": 1710, "菊池寛": 83, "鈴木三重吉": 1671,
}
_AOZORA_WORK = re.compile(r'href="\.\./cards/(\d+)/card(\d+)\.html"[^>]*>([^<]+)</a>')
_AOZORA_HTMLFILE = re.compile(r'href="\./files/(\d+_\d+\.html)"')


def aozora_author_works(person_id: int, limit: int = 40) -> list[tuple[str, str]]:
    """[(title, work_html_url), ...] for one Aozora author, resolving each work
    card to its XHTML file."""
    list_url = f"https://www.aozora.gr.jp/index_pages/person{person_id}.html"
    try:
        raw = _get_bytes(list_url, AOZORA_UA, "text/html")
    except (NetworkBudgetExceeded, Exception):
        return []
    html = raw.decode("shift_jis", errors="replace")
    out = []
    for card_person, card_id, title in _AOZORA_WORK.findall(html)[:limit]:
        card_url = f"https://www.aozora.gr.jp/cards/{int(card_person):06d}/card{card_id}.html"
        try:
            card = _get_bytes(card_url, AOZORA_UA, "text/html").decode(
                "shift_jis", errors="replace")
        except (NetworkBudgetExceeded, Exception):
            continue
        m = _AOZORA_HTMLFILE.search(card)
        if m:
            out.append((title.strip(),
                        f"https://www.aozora.gr.jp/cards/{int(card_person):06d}/files/{m.group(1)}"))
    return out


def fetch_kernel(network: int = 20) -> list[JapaneseText]:
    WEB_CACHE.set_network_budget(network)
    out = []
    for title in kernel_titles():
        if WEB_CACHE.remaining_network_budget() == 0:
            break
        story = fetch(title)
        if story:
            out.append(story)
    return out


def main() -> None:
    import json
    for story in fetch_kernel():
        print(json.dumps({"title": story.title, "url": story.url,
                          "japanese_chars": story.japanese_char_count,
                          "head": story.text[:80]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
