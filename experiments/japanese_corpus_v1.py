#!/usr/bin/env python3
"""Fetch Japanese children's stories for the developmental reading curriculum.

Read-only, cached, budget-aware (same ReadOnlyWebCache the English path uses).
Source: ja.wikisource.org -- folktale retellings (楠山正雄 etc.), translated
Grimm/Andersen/Aesop, and the pre-war graded primers (尋常小学読本).  Text comes
back as plain UTF-8; ruby and editorial headers are stripped.

`KERNEL` is a hand-picked seed of the simplest works, needed to bootstrap from
~zero known Japanese vocabulary before autonomous discovery can take over.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

from web_cache import WEB_CACHE, NetworkBudgetExceeded

API = "https://ja.wikisource.org/w/api.php"
UA = "AI_Noise/0.30 (developmental Japanese reading; read-only)"

# Bootstrap kernel: simplest first.  Titles verified to exist on ja.wikisource.
KERNEL = (
    "猿蟹合戦", "舌切り雀", "桃太郎 (楠山正雄)", "花咲かじじい", "かちかち山",
    "こぶとり", "おむすびころりん", "浦島太郎 (楠山正雄)", "一寸法師 (楠山正雄)",
    "ネズミの嫁入り", "きつねと葡萄", "北風と太陽", "うさぎと亀", "アリとキリギリス",
)

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
        data = WEB_CACHE.get_json(f"{API}?{params}", UA)
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
        data = WEB_CACHE.get_json(f"{API}?{params}", UA)
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


def fetch_aozora(html_url: str) -> JapaneseText | None:
    """A work's XHTML file on aozora.gr.jp (Shift_JIS), ruby readings dropped."""
    try:
        raw = WEB_CACHE.get_bytes(html_url, AOZORA_UA, "text/html")
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
    title = ""
    m = re.search(r'<meta name="DC.Title" content="([^"]+)"', html)
    if m:
        title = m.group(1)
    return JapaneseText(title=title or html_url.rsplit("/", 1)[-1], url=html_url, text=text)


def category_members(category: str, limit: int = 100) -> list[str]:
    """Titles in a ja.wikisource category (e.g. 'Category:童話')."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "categorymembers", "cmtitle": category,
        "cmnamespace": 0, "cmlimit": limit, "cmtype": "page",
        "format": "json", "formatversion": 2})
    try:
        data = WEB_CACHE.get_json(f"{API}?{params}", UA)
    except (NetworkBudgetExceeded, Exception):
        return []
    return [item["title"] for item in data.get("query", {}).get("categorymembers", [])]


def fetch_kernel(network: int = 20) -> list[JapaneseText]:
    WEB_CACHE.set_network_budget(network)
    out = []
    for title in KERNEL:
        story = fetch(title)
        if story is None:
            for alt in search(title.split(" (")[0], 3):
                story = fetch(alt)
                if story:
                    break
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
