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


# Standalone national-reader stories on ja.wikisource (Category:国定教科書) that
# are genuine graded elementary material -- short narratives, roughly level 2-3.
_SCHOOL_READER_SEED = ("初等科國語 一/電車", "汽車 (アサヒ読本)", "久田船長",
                       "初等科國語 三/濱田彌兵衛")


def school_reader_titles(limit: int = 30) -> tuple[str, ...]:
    """Pages in Category:国定教科書 -- the prewar national elementary readers.
    Prefer the 國語 readers (not 國史 history) and the lower grades."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "categorymembers", "cmtitle": "Category:国定教科書",
        "cmnamespace": 0, "cmlimit": limit, "cmtype": "page",
        "format": "json", "formatversion": 2})
    try:
        data = _get_json(f"{API}?{params}")
    except (NetworkBudgetExceeded, Exception):
        return _SCHOOL_READER_SEED
    def grade(t: str) -> int:
        for g, n in (("一", 1), ("二", 2), ("三", 3), ("四", 4),
                     ("五", 5), ("六", 6), ("七", 7), ("八", 8)):
            if f"國語 {g}/" in t or f"國語{g}/" in t:
                return n
        return 4
    titles = [m["title"] for m in data.get("query", {}).get("categorymembers", [])
              if "國史" not in m["title"] and "憲法" not in m["title"]]
    titles.sort(key=grade)
    return tuple(dict.fromkeys(_SCHOOL_READER_SEED + tuple(titles)))


def kernel_titles(limit: int = 40) -> tuple[str, ...]:
    return KERNEL_SEED + aesop_kernel(limit) + school_reader_titles()


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
# 旧字体 -> 新字体: common kyūjitai in pre-1946 texts (国定教科書, older Aozora
# cards).  Folded so the parser's vocabulary and the RNN see one script.
_KYUJITAI_OLD = "國來學會觀廣圓兒澤濱樂讀聲晝賣對舊眞氣拂惠應歸當團圖縣靜藝廳假缺齒醫邊鐵驛發稱續戀螢濟參號單點禮營勞區收萬與實黨轉體莊藏經"
_KYUJITAI_NEW = "国来学会観広円児沢浜楽読声昼売対旧真気払恵応帰当団図県静芸庁仮欠歯医辺鉄駅発称続恋蛍済参号単点礼営労区収万与実党転体荘蔵経"
assert len(_KYUJITAI_OLD) == len(_KYUJITAI_NEW)
_KYUJITAI = str.maketrans(_KYUJITAI_OLD, _KYUJITAI_NEW)
_OLD_KANA_SUB = [
    (re.compile(r"くわ"), "か"), (re.compile(r"ぐわ"), "が"),
    (re.compile(r"ぢ"), "じ"), (re.compile(r"づ"), "ず"),
    # sokuon written つ: 言つた / 買つて / なつたので -- not the stem of
    # 伝える(つたえる) / 伝えて / 伝えない
    (re.compile(r"([ぁ-ゖ㐀-鿿])つ([たて])(?!え[るてられ]|ない)"), r"\1っ\2"),
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
    (re.compile(r"おほ(?=きい|きな|く|い|ぜい|かた|やけ|むね)"), "おお"),   # おほきい -> おおきい
    (re.compile(r"かほ(?=[。、」\s]|$|を|は|に)"), "かお"),                # おかほ -> おかお
    (re.compile(r"ゐ"), "い"),
    (re.compile(r"([ぁ-ゖ])ふ(?=[、。」]|よ|ね|わ|の)"), r"\1う"),      # 似合ふよ -> 似合うよ
]


def _dekana(text: str) -> str:
    """Fold 歴史的仮名遣い to modern kana so the parser and RNN see one
    orthography.  Applied to Aozora children's texts (pre-1946)."""
    marks = sum(text.count(m) for m in (
        "ゐ", "ゑ", "なつた", "だつた", "あつた", "いつた", "さう", "やう", "ませう",
        "でせう", "ひました", "ひます", "おほき", "おほい", "おほく", "であふ",
        "らう。", "ちやん", "しませう", "きらひ", "こひ", "おもひ"))
    if marks < 4:
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
    return _dekana(text.translate(_KYUJITAI))


@dataclass
class JapaneseText:
    title: str
    url: str
    text: str

    @property
    def japanese_char_count(self) -> int:
        return len(JP_CHARS.findall(self.text))


def _strip_section_headings(text: str) -> str:
    """A short line with no sentence punctuation, on its own between blank lines
    (or at the very start), is an Aozora section heading -- "生い立ち" glued to the
    first sentence otherwise becomes part of its subject."""
    text = re.sub(r"\A(?:[^\n。！？、」』]{1,18}\n)+", "", text)
    return re.sub(r"\n[ \t　]*\n[^\n。！？、」』]{1,18}\n[ \t　]*\n", "\n\n", text)


def _clean(raw: str) -> str:
    text = _NOTES.sub("", raw)
    text = _HEADING.sub("", text)
    text = _RUBY.sub("", text)
    text = _BRACKET.sub("", text)
    text = _strip_section_headings(text)
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
    text = _strip_section_headings(text)
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


# --- Tatoeba: a large pool of short, clean, single-clause sentences ----------
# The Aozora / wikisource corpus is literary prose that the from-scratch parser
# mangles.  Tatoeba is CC-BY 2.0 FR example sentences -- mostly simple, one
# clause, which the parser handles far better.  Used to build "graded readers":
# a bundle of ~20 sentences near a target level acts as one shelf entry.  These
# are NOT narratives -- no protagonist / order / retelling -- so they are graded
# by parse quality + vocabulary and kept out of the narrative benchmarks.
TATOEBA_SENTENCES_URL = (
    "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2")
TATOEBA_LICENSE = "CC-BY-2.0-FR"
TATOEBA_ATTRIBUTION = ("Tatoeba (https://tatoeba.org) -- example sentences by its "
                       "contributors, released under CC-BY 2.0 FR")
_KANJI = re.compile(r"[㐀-鿿]")
_LATIN_DIGIT = re.compile(r"[A-Za-z0-9０-９]")


def _sentence_level(sentence: str) -> float:
    """Cheap per-sentence difficulty: kana + short ~ 1.4, kanji-dense + long ~ 4+."""
    kanji = len(_KANJI.findall(sentence))
    return round(min(6.0, 1.4 + 0.18 * kanji + 0.04 * max(0, len(sentence) - 12)), 2)


def _tatoeba_lines() -> list[str]:
    """The decompressed jpn_sentences.tsv, one cached bulk download."""
    import bz2
    try:
        raw = _get_bytes(TATOEBA_SENTENCES_URL, UA, "application/octet-stream")
        return bz2.decompress(raw).decode("utf-8", "replace").splitlines()
    except (NetworkBudgetExceeded, Exception):
        return []


# process-lifetime cache of the parse-filtered pool per (level, band) -- the
# download is cached on disk but decompressing + parse-filtering 200k lines is
# slow, and `_fetch_more_books` runs every few cycles in one long-lived process
_CLEAN_POOL: "dict[tuple[float, float], list[tuple[int, str]]]" = {}


def _tatoeba_clean_pool(level: float, band: float, network: int,
                        _lines: "list[str] | None" = None) -> list[tuple[int, str]]:
    key = (round(level, 1), band)
    if key in _CLEAN_POOL:
        return _CLEAN_POOL[key]
    WEB_CACHE.set_network_budget(network)
    lines = _lines if _lines is not None else _tatoeba_lines()
    if not lines:
        return []
    lo, hi = level - band, level + band
    cand: list[tuple[int, str]] = []
    for line in lines:
        p = line.split("\t")
        if len(p) < 3:
            continue
        try:
            sid = int(p[0])
        except ValueError:
            continue
        s = p[2].strip()
        if (not (6 <= len(s) <= 28) or not s.endswith(("。", "！", "？"))
                or _LATIN_DIGIT.search(s) or "「" in s):
            continue
        if lo <= _sentence_level(s) <= hi:
            cand.append((sid, s))
        if len(cand) >= 2000:
            break
    cand.sort()
    from japanese_event_v1 import extract_story
    clean = [(sid, s) for sid, s in cand
             if (evs := extract_story(_modernise(s)))
             and evs[0].subject and evs[0].verb and evs[0].subject_explicit]
    if _lines is None:                       # only cache real (not test) data
        _CLEAN_POOL[key] = clean
    return clean


def tatoeba_readers(level: float, n_readers: int = 6, per_reader: int = 20,
                    band: float = 0.4, network: int = 4, skip: int = 0,
                    _lines: "list[str] | None" = None) -> list[JapaneseText]:
    """`n_readers` bundles of `per_reader` short Tatoeba sentences within `band`
    of `level`, deterministic by sentence id, after skipping the first `skip`
    clean sentences (so successive passes get fresh material).  Each bundle is a
    JapaneseText whose url is `tatoeba://reader/<level>/<first id>`.  Only
    sentences the heuristic parser turns into a clean (explicit subject + verb)
    event are kept -- a reader is made of sentences Noise can actually read."""
    clean = _tatoeba_clean_pool(level, band, network, _lines=_lines)[skip:]
    readers = []
    for i in range(0, len(clean) - per_reader + 1, per_reader):
        chunk = clean[i:i + per_reader]
        first = chunk[0][0]
        readers.append(JapaneseText(
            title=f"やさしい文集 lv{level:.1f} #{first}",
            url=f"tatoeba://reader/{level:.1f}/{first}",
            text="\n".join(s for _, s in chunk)))
        if len(readers) >= n_readers:
            break
    return readers


def main() -> None:
    import json
    for story in fetch_kernel():
        print(json.dumps({"title": story.title, "url": story.url,
                          "japanese_chars": story.japanese_char_count,
                          "head": story.text[:80]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
