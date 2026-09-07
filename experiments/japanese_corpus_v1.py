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
KERNEL_SEED = ("猿蟹合戦", "独逸童話集", "お伽噺")
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
_HIRA = re.compile(r"[ぁ-ゖ]")
_KATA = re.compile(r"[ァ-ヶ]")


# 歴史的仮名遣い -> 現代仮名遣い: regular, bounded rules.  The risky word-medial
# はひふへほ -> わいうえお is applied only in clear verb/adjective contexts (after
# a kanji or before an inflection), never to the は / へ particles.
_OLD_KANA_TABLE = str.maketrans({"ゐ": "い", "ゑ": "え", "ヰ": "イ", "ヱ": "エ",
                                 "ゔ": "ぶ"})
_OLD_KANA_SUB = [
    (re.compile(r"くわ"), "か"), (re.compile(r"ぐわ"), "が"),
    (re.compile(r"ぢ"), "じ"), (re.compile(r"づ"), "ず"),
    # sokuon written つ: 言つた / 待つて / ぶつかつた
    (re.compile(r"([ぁ-ゖ㐀-鿿])つ(?=[たてちゃ])"), r"\1っ"),
    (re.compile(r"ちやん"), "ちゃん"),                     # 〜ちやん -> 〜ちゃん
    # au / iu / eu -> ou / yuu / you  (さう->そう, ませう->ましょう, てふ->ちょう)
    (re.compile(r"(せ|でせ|ませ|ましせ)う"), r"\1ょう"),
    (re.compile(r"やう"), "よう"), (re.compile(r"さう"), "そう"),
    (re.compile(r"かう(?=[。、」\s]|$)"), "こう"), (re.compile(r"だら?う"), "だろう"),
    (re.compile(r"てふ"), "ちょう"), (re.compile(r"けふ"), "きょう"),
    # verb endings: 〜ひ / 〜ひます / 〜ふ  (思ひ, 買ふ, 言ひました)
    (re.compile(r"ひ(?=(まし|ます|なさ|た|て|、|。|」))"), "い"),
    (re.compile(r"([ぁ-ゖ])ふ(?=[。、」\s]|$)"), r"\1う"),
    (re.compile(r"([かあこそのま])は(?=[。、」\s]|$)"), r"\1わ"),   # かは(川). 〜は particle excluded
    (re.compile(r"いへ(?=ば|、|。)"), "いえ"),
]


def _dekana(text: str) -> str:
    """Fold 歴史的仮名遣い to modern kana so the parser and RNN see one
    orthography.  Applied to Aozora children's texts (pre-1946)."""
    marks = sum(text.count(m) for m in ("ゐ", "ゑ", "なつた", "つた。", "さう", "ひました",
                                        "ひます", "やう", "ませう"))
    if marks < 3:
        return text
    text = text.translate(_OLD_KANA_TABLE)
    for pattern, repl in _OLD_KANA_SUB:
        text = pattern.sub(repl, text)
    return text


def _modernise(text: str) -> str:
    """Old children's texts are set entirely in katakana (一ピキノイヌガ…) or in
    歴史的仮名遣い (なつた, おもひ, ゐる).  When katakana dominates it is
    orthography, not loanwords: fold it to hiragana.  Then fold historical kana
    so the parser and RNN see one modern orthography."""
    kata, hira = len(_KATA.findall(text)), len(_HIRA.findall(text))
    if kata >= 12 and kata > hira * 1.3:
        text = _KATA.sub(lambda m: chr(ord(m.group()) - 0x60), text)
    return _dekana(text)


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
    return _modernise(text)


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
    text = _modernise(re.sub(r"[ \t　]+", "", re.sub(r"\n{2,}", "\n", text)).strip())
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


# Aozora author "person" pages -> their public-domain works.  Person IDs are
# from list_person_all_extended_utf8.csv; the earlier 浜田広介 1710 / 鈴木三重吉
# 1671 were wrong (a German poet, and 津田黄昏).  Ordered easiest first: folktale
# retellings and 幼年童話, then 童謡, then the more literary children's authors.
AOZORA_AUTHORS = {
    "楠山正雄": 329,        # 日本の folktale retellings, 敬体, ~300 works
    "村山籌子": 1172,       # 幼年童話 -- the simplest
    "新美南吉": 121,
    "巌谷小波": 981,        # お伽噺の祖 -- 日本昔噺 / 世界お伽噺
    "佐々木喜善": 263,      # 聴耳草紙 -- Tōno folk tales, short and plain
    "鈴木三重吉": 107,      # 赤い鳥, 古事記物語
    "宮原晃一郎": 809,      # Andersen / Nordic tale translations
    "北原白秋": 106,        # 童謡 -- very short
    "野口雨情": 286,        # 童謡
    "秋田雨雀": 1584,       # 童話劇
    "小川未明": 1475,
    "浜田広介": 1054,       # some works only; 廣介 kanji varies
    "夢野久作": 96,         # 蟻のおれい etc. children's pieces
    "有島武郎": 25,         # 一房の葡萄
    "豊島与志雄": 906,
    "芥川竜之介": 879,      # 蜘蛛の糸 / 杜子春 / アグニの神
    "菊池寛": 83,
    "宮沢賢治": 81,
}
# ja.wikisource pages / collections worth trying beyond イソップ童話集
_WIKISOURCE_EXTRA = ("独逸童話集", "お伽噺", "こがね丸", "花咲爺", "舌切雀",
                     "かちかち山", "文福茶釜", "浦島太郎", "一寸法師")
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
