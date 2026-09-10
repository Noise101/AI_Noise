#!/usr/bin/env python3
"""Learn what Japanese words MEAN as revisable beliefs.

The reading loop's "comprehension" is structural: it never represents what a
word denotes.  This module closes that gap.  Every meaning is a *belief* with a
confidence and a source trail, and it can be revised:

  * distributional semantics -- for every content word Noise reads (its own
    heuristic (subject, obj, verb) tokens), accumulate the content words it
    co-occurs with and a usage PROFILE (does it act, or is it acted on?).
  * testimony -- ja.wiktionary / ja.wikipedia give a genus ("...イヌ科に属する
    哺乳動物" -> 哺乳動物); the optional local model, asked a closed-set
    discrimination question, gives a coarse class.  Testimony enters a belief at
    a *capped* weight (ARCHITECTURE.md invariant 10: a local model / dictionary
    has no vote) -- it is a hypothesis, believed provisionally, never sufficient
    on its own.
  * evidence -- Noise's OWN reading: how the word is used (acts like a creature /
    is handled like a thing / is a destination like a place), and the classes of
    the words it co-occurs with that Noise already understands.  Evidence is what
    moves confidence and what a word must have before it counts as "understood".
  * revision -- each cycle the belief is recomputed.  When reading evidence
    contradicts an earlier testimony-only genus the belief flips and the change
    is kept in `revisions`.  A word "understood" from testimony that later
    reading disagrees with loses that status.

Capability (frozen, held-out): a fixed set of words is NEVER researched and NEVER
asked to the local model.  Noise must classify them from reading + beliefs it
built on OTHER words.  Score = does the belief machinery put a held-out word in
the right coarse class (vs its ja.wiktionary reference) more often than naive
co-occurrence propagation -- one-sided significance over per-word gains.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request
from collections import Counter

from web_cache import WEB_CACHE, NetworkBudgetExceeded

VERSION = 2                   # belief-revision layer
SELECTION_VERSION = 4         # held-out set is now built lazily, genus-validated
USER_AGENT = "AI_Noise/0.32 (developmental Japanese word meaning; read-only)"
WIKTIONARY_API = "https://ja.wiktionary.org/w/api.php"
WIKIPEDIA_API = "https://ja.wikipedia.org/w/api.php"
HELD_OUT_SALT = "japanese-word-meaning:heldout:v1"
HELD_OUT_FRACTION = 5          # ~1/5 of the vocabulary is the frozen test set
RESEARCH_PER_CYCLE = 3         # ja.wiktionary fetches per cycle (train words only)
LLM_ASKS_PER_CYCLE = 1         # local-model discrimination questions (train words only)
MIN_TEST_WORDS = 8
SIGNIFICANCE_Z = 3.0
EXPLAIN_TERMS = 5
TEST_SET_TARGET = 40          # held-out set grows to this, genus-validated
TEST_PROBES_PER_CYCLE = 2     # held-out candidates genus-checked per cycle

# role / abstract nouns that dominate a literary corpus but have no concrete
# denotation to learn -- kept out of the held-out capability set
_ABSTRACT_ROLE = frozenset((
    "老人", "若者", "青年", "少年", "少女", "娘", "息子", "父親", "母親", "両親",
    "お母さん", "お父さん", "おかあさん", "おとうさん", "にいさん", "ねえさん",
    "奥さん", "奥さま", "主人", "夫人", "人たち", "人間", "人物", "人々", "者",
    "子供たち", "子どもたち", "私たち", "僕たち", "自分たち", "みなさん", "諸君",
    "言葉", "仕方", "仕事", "様子", "気持ち", "考え", "意味", "理由", "問題",
    "沈黙", "運命", "自由", "幸福", "不幸", "真実", "事実", "現実", "世界",
    "生活", "人生", "時間", "時代", "瞬間", "場合", "状態", "関係", "方法",
    "以上", "以下", "以前", "以後", "全体", "部分", "多く", "初め", "終り",
    "はじめ", "おわり", "うしろ", "まえ", "あいだ", "なか", "そば", "ほう"))

TESTIMONY_CAP = 0.35          # testimony alone can never push confidence past this
EVIDENCE_HEADROOM = 0.65      # reading evidence fills the rest
UNDERSTOOD_CONF = 0.60        # confidence at/above this ...
UNDERSTOOD_EVIDENCE = 0.25    # ... AND this much independent reading evidence

_CONTENT = re.compile(r"^[ぁ-ゟ゠-ヿ一-鿿]{2,}$")
_STOP = frozenset((
    "こと", "もの", "ところ", "とき", "ため", "よう", "そう", "これ", "それ", "あれ",
    "どれ", "だれ", "なに", "ここ", "そこ", "あそこ", "どこ",
    "わたし", "わたしたち", "あなた", "みんな", "みな", "ひとつ", "ふたり",
    "かれ", "かのじょ", "ぼく", "おれ", "じぶん", "自分", "彼", "彼女", "私",
    "ひと", "人", "人々", "男", "女", "子", "子供", "こども", "坊ちゃん", "お前", "おまえ",
    "する", "なる", "ある", "いる", "くる", "いく", "みる", "いう", "おもう",
    "できる", "しまう", "いた", "した", "きた"))
# MediaWiki / dictionary furniture that must never be read as a word's genus
_BAD_GENUS = frozenset((
    "日本語", "編集", "参照", "同音異義語", "和語", "類義語", "対義語", "翻訳", "発音",
    "未然形", "連用形", "終止形", "連体形", "仮定形", "命令形", "活用", "語源", "成句",
    "であるもの", "もの", "こと", "使い方", "熟語", "表記", "字源", "異体字"))
# genus: the head noun of a defining sentence -- the trailing >=2-char
# kanji/katakana run (plus a short okurigana) right before である / 。 / のこと.
# "...イヌ科に属する哺乳動物。" -> 哺乳動物 ; "...時刻を示す道具。" -> 道具.
_GENUS = re.compile(
    r"([一-鿿゠-ヿ]{2,}[ぁ-ゟ]{0,2})"
    r"(?:である|です|。|のこと|を指す|の(?:一種|総称|名))")

# --- coarse ontology ---------------------------------------------------------
# A small closed set of everyday classes.  Both the fine dictionary genus and
# the local model's answer are folded onto this so beliefs from different
# sources can agree or conflict.
COARSE_CLASSES = ("生き物", "人", "植物", "食べ物", "道具", "場所",
                  "自然物", "出来事", "気持ち")
_COARSE_RULES = (
    (("哺乳", "動物", "獣", "けもの", "鳥", "とり", "魚", "さかな", "虫", "昆虫",
      "類", "犬", "猫", "狐", "狸", "猿", "熊", "鼠", "兎", "馬", "牛", "羊",
      "蛙", "亀", "蛇", "蝙蝠", "鶴", "烏", "鳩", "蟻", "生物"), "生き物"),
    (("人物", "人間", "者", "士", "王", "女王", "神", "妖精", "巨人", "小人",
      "少年", "少女", "老人", "武士", "きこり", "登場人物", "職業", "家来"), "人"),
    (("植物", "花", "木", "草", "樹", "野菜", "きのこ", "苔"), "植物"),
    (("食べ物", "食物", "料理", "菓子", "果実", "果物", "飲み物", "穀物",
      "パン", "実", "米", "飯"), "食べ物"),
    (("道具", "器具", "器", "具", "機械", "武器", "乗り物", "車", "船", "衣類",
      "服", "家具", "容器", "楽器", "品", "物品", "貨幣", "金銭"), "道具"),
    (("場所", "地域", "地方", "土地", "国", "町", "村", "都市", "山", "川",
      "海", "湖", "島", "森", "林", "建物", "部屋", "道路", "施設", "空間"), "場所"),
    (("天体", "星", "太陽", "月", "自然現象", "鉱物", "石", "岩", "水", "火",
      "風", "雲", "雪", "光", "元素", "物質"), "自然物"),
    (("現象", "出来事", "事件", "儀式", "行事", "戦い", "戦争", "行為",
      "動作", "祭り"), "出来事"),
    (("感情", "気持ち", "心情", "情動"), "気持ち"),
)


def _coarse(fine: str) -> str:
    """Fold a fine genus / free-text class onto the closed coarse ontology."""
    if not fine:
        return ""
    if fine in COARSE_CLASSES:
        return fine
    for keys, cls in _COARSE_RULES:
        if any(k in fine for k in keys):
            return cls
    return ""


def _compatible(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return {a, b} == {"人", "生き物"}          # a person is a creature


_GLUED_PARTICLE = re.compile(r"^(と|て|は|も|に|を|が|で|の|や)([一-鿿゠-ヿぁ-ゟ]{2,})$")


def _norm(w: str) -> str:
    w = (w or "").strip()
    # the parser glues a quotative/comitative particle to a following PRONOUN in
    # dialogue ("と彼" -> "とかれ", "とわたし").  Strip it only then -- real nouns
    # like とけい(時計) / となり(隣) keep their と.
    m = _GLUED_PARTICLE.match(w)
    if m and m.group(2) in _STOP:
        return m.group(2)
    return w


def _is_content(w: str) -> bool:
    w = _norm(w)
    return bool(_CONTENT.match(w)) and w not in _STOP


def _is_wordlike(w: str) -> bool:
    """A single word, not a phrase or a parse fragment: 2..6 chars, no の / 、;
    no dangling case particle on a LONGER run (はと/ひと/あと are real words);
    not と+katakana-name."""
    w = _norm(w)
    return (_is_content(w) and 2 <= len(w) <= 6 and "の" not in w and "、" not in w
            and not (len(w) >= 4 and w.endswith(("は", "が", "を", "に", "で")))
            and not re.match(r"^と[゠-ヿ]", w))


def _held_out(word: str) -> bool:
    h = hashlib.sha256(f"{HELD_OUT_SALT}:{word}".encode()).hexdigest()
    return int(h[:8], 16) % HELD_OUT_FRACTION == 0


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[ぁ-ゟ゠-ヿ一-鿿]{2,}", text or "") if _is_content(w)]


_KANJI_JP = re.compile(r"^[ぁ-ゟ゠-ヿ一-鿿々〆ヶ]+$")
# simplified-Chinese chars that leak in from ja.wiktionary's 中国語 sections
_CN_ONLY = set("时间说话见这个们来对开关会门电脑页务实际标边过还爱国车轮马鸟鱼语学")
_GENUS_JUNK = frozenset(("三省堂", "岩波", "広辞苑", "現世", "嗅覚システム",
                         "抽象概念", "総体", "一切", "存在物"))


def _plausible_genus(g: str) -> bool:
    """A genus is a plain everyday noun -- 2..6 chars, JIS-range, not a fragment."""
    return (bool(g) and 2 <= len(g) <= 6 and g not in _BAD_GENUS and g not in _STOP
            and g not in _GENUS_JUNK and bool(_KANJI_JP.match(g))
            and not (set(g) & _CN_ONLY)
            and not g.endswith(("など", "こと", "もの", "え", "り")))


# --- usage profile: how Noise sees the word used ---------------------------
# The heuristic parser emits noisy, often multi-token, kanji-or-kana verb
# strings ("持って来て取りおろさす").  Match on a stem *substring* and include
# both scripts.  Behaviour that needs a mind (perceive / speak / feel) marks a
# creature; motion alone does not (a train moves).
_ANIMATE_VERBS = ("みる", "見", "きく", "聞", "かぐ", "嗅", "いう", "言", "はなす", "話",
                  "こたえ", "答え", "たずね", "尋ね", "よぶ", "呼", "なく", "泣",
                  "ほえ", "吠", "わら", "笑", "おこ", "怒", "おもう", "思", "かんがえ",
                  "考え", "ねむ", "眠", "ねる", "寝", "おき", "起", "たべ", "食べ",
                  "のむ", "飲", "くう", "食う", "はたらく", "働", "あそ", "遊", "うたい",
                  "歌", "すわ", "座", "にげ", "逃")
_MOTION_VERBS = ("あるく", "歩", "はしる", "走", "とぶ", "飛", "およ", "泳", "くる", "来",
                 "いく", "行", "かえ", "帰", "つく", "着", "とまる", "止ま", "すすむ",
                 "進", "のぼ", "登", "上", "おり", "降", "でる", "出", "はいる", "入",
                 "うごく", "動", "きえる", "消え", "ながれ", "流れ")
_HANDLE_VERBS = ("つくる", "作", "もつ", "持", "つかう", "使", "なげ", "投げ", "とる",
                 "取", "かう", "買", "うる", "売", "こわ", "壊", "わる", "割", "きる",
                 "切", "ひらく", "開", "あけ", "開け", "しめ", "閉", "はこぶ", "運",
                 "おく", "置", "ひろう", "拾", "みがく", "磨", "ひく", "引", "つかむ",
                 "掴", "にぎ", "握", "むけ", "向け", "かつ", "担", "ならす", "鳴らす",
                 "ふく", "吹", "あて", "当て", "よせ", "寄せ", "すすめ", "与え", "召")
_EAT_VERBS = ("たべ", "食べ", "のむ", "飲", "くう", "食う", "かじ", "齧", "あじわ", "味わ")
_WEAR_VERBS = ("きる", "着", "はく", "穿", "かぶ", "被", "まと", "纏", "めす", "召")


def _verb_has(verb: str, stems: "tuple[str, ...]") -> bool:
    v = _norm(verb)
    return any(s in v or s[:2] in v for s in stems if len(s) >= 2) \
        or any(s in v for s in stems if len(s) == 1)


def _reading_class(profile: dict) -> "tuple[str, float]":
    """A coarse class inferred purely from how Noise saw the word used, plus a
    0..1 strength.  Empty when the usage is uninformative."""
    if not profile:
        return "", 0.0
    subj = profile.get("subj", 0)
    obj = profile.get("obj", 0)
    sv = Counter(profile.get("subj_verbs", {}))
    ov = Counter(profile.get("obj_verbs", {}))
    total = subj + obj
    if total < 3:
        return "", 0.0
    animate = sum(n for v, n in sv.items() if _verb_has(v, _ANIMATE_VERBS))
    hunted = sum(n for v, n in ov.items() if _verb_has(v, ("かり", "狩", "つかまえ", "捕")))
    motion = sum(n for v, n in sv.items() if _verb_has(v, _MOTION_VERBS))
    handled = sum(n for v, n in ov.items() if _verb_has(v, _HANDLE_VERBS))
    eaten = sum(n for v, n in ov.items() if _verb_has(v, _EAT_VERBS))
    worn = sum(n for v, n in ov.items() if _verb_has(v, _WEAR_VERBS))
    suru = sum(n for v, n in ov.items() if _norm(v) in ("する", "した", "して行る"))
    obj_ratio = obj / total

    if eaten and eaten >= handled and eaten >= animate:
        return "食べ物", min(1.0, 0.3 + eaten / total)
    if animate + hunted and animate + hunted >= handled:
        # perceives / speaks / feels / is hunted -> a creature (person is a
        # sub-case that testimony/taxonomy refines)
        return "生き物", min(1.0, 0.3 + (animate + hunted) / total)
    if worn and obj_ratio >= 0.5:
        return "道具", min(1.0, 0.4 + worn / total)          # clothing
    if obj_ratio >= 0.65 and handled:
        return "道具", min(1.0, 0.35 + handled / total)      # a handled thing
    if motion and handled and not animate:
        return "道具", min(0.8, 0.35 + handled / total)      # moves + handled = vehicle
    if suru and subj <= total * 0.2 and not handled:
        return "出来事", min(0.8, 0.3 + suru / total)        # X をする -> an act
    if obj_ratio >= 0.6 and not (animate or motion or eaten):
        return "道具", 0.3                                   # only ever acted on
    return "", 0.0


# --- ja.wiktionary (CC-BY-SA) ---------------------------------------------
def _wiktionary_gist(word: str) -> dict | None:
    """{'genus': str, 'terms': [str], 'related': [str]} from the 名詞 definition."""
    try:
        from story_web_curriculum_v13 import _TextExtractor
        params = urllib.parse.urlencode({
            "action": "parse", "page": word, "prop": "text",
            "format": "json", "formatversion": 2})
        parsed = WEB_CACHE.get_json(f"{WIKTIONARY_API}?{params}", USER_AGENT).get("parse", {})
    except (NetworkBudgetExceeded, Exception):
        return None
    ex = _TextExtractor()
    ex.feed(parsed.get("text", ""))
    raw = ex.text().replace(" ", "")
    # strip MediaWiki section furniture: ".[編集]." markers, bare "." separators
    text = re.sub(r"\[編集\]", "", raw)
    text = re.sub(r"[.．]{1,}", "。", text)
    text = re.sub(r"[（(][^）)]{0,20}[)）]", "", text)     # (冬の季語) etc.
    if len(text) < 20:
        return None
    # the definitions between the 名詞/動詞 heading and the next heading
    body = text
    m = re.search(r"(名詞|動詞|形容詞)。(.{6,400}?)(?:成句|関連語|翻訳|発音|活用|語源|類義語|$)", text)
    if m:
        body = m.group(2)
    # first real definition sentence: skip navigation ("日本語", "〜も参照") and
    # the headword gloss line ("きつね【狐】")
    sentences = [s for s in re.split(r"。", body) if len(s) >= 6]
    defsent = ""
    for s in sentences[:5]:
        if ("参照" in s or "同音異義" in s or "【" in s or s.strip() in _BAD_GENUS
                or "漢字表記" in s or "異表記" in s or s.endswith("形") and len(s) < 12):
            continue
        defsent = s + "。"
        break
    if not defsent:
        return None
    genus = ""
    for g in _GENUS.finditer(defsent):
        cand = g.group(1)
        if _plausible_genus(cand) and word not in cand:
            genus = cand
    rel_m = re.search(r"関連語。(.{0,120})", text)
    related = [w for w in _content_words(rel_m.group(1)) if w not in _BAD_GENUS][:8] if rel_m else []
    terms = [w for w in _content_words(defsent) if w != word and w not in _BAD_GENUS][:12]
    if not genus and not terms and not related:
        return None
    if not genus:
        genus = _wikipedia_genus(word)
    return {"genus": genus, "terms": terms, "related": related[:8]}


def _wikipedia_genus(word: str) -> str:
    """The head noun of a ja.wikipedia article's first sentence
    ("アラジンは、『千夜一夜物語』の登場人物。" -> 登場人物)."""
    try:
        params = urllib.parse.urlencode({
            "action": "query", "titles": word, "redirects": 1, "prop": "extracts",
            "exintro": 1, "explaintext": 1, "exsentences": 1,
            "format": "json", "formatversion": 2})
        pages = WEB_CACHE.get_json(f"{WIKIPEDIA_API}?{params}", USER_AGENT
                                  ).get("query", {}).get("pages", [])
    except (NetworkBudgetExceeded, Exception):
        return ""
    if not pages or pages[0].get("missing"):
        return ""
    lead = (pages[0].get("extract") or "").replace(" ", "")
    if "曖昧さ回避" in lead or not lead:
        return ""
    for g in _GENUS.finditer(lead.split("。")[0] + "。"):
        c = g.group(1)
        if _plausible_genus(c) and word not in c:
            return c
    return ""


# --- optional local model: a closed-set discrimination question -----------
class _LLMClient:
    """Minimal Ollama client.  Asks ONE closed-set question and accepts only an
    answer inside the coarse ontology -- anything else is discarded.  The answer
    is testimony (evidence score 0), never authority."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3.8:27b")

    def available(self) -> bool:
        if os.environ.get("AI_NOISE_SKIP_LOCAL_LLM"):
            return False
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as r:
                models = json.load(r).get("models", [])
            return any(m.get("name") == self.model for m in models)
        except Exception:
            return False

    def ask_class(self, word: str) -> str | None:
        options = "／".join(COARSE_CLASSES)
        prompt = (f"日本語の単語「{word}」が指すものは、次のどれに一番近いですか。"
                  f"リストの語をそのまま一つだけ答えてください：{options}")
        schema = {"type": "object",
                  "properties": {"class": {"type": "string", "enum": list(COARSE_CLASSES)}},
                  "required": ["class"]}
        payload = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False, "think": False,
            "format": schema, "options": {"temperature": 0.0, "num_predict": 12}}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                answer = json.loads(json.load(r).get("response", "{}")).get("class", "")
        except Exception:
            return None
        answer = str(answer).strip()
        return answer if answer in COARSE_CLASSES else None


# --- learning + belief revision -----------------------------------------
def _blank() -> dict:
    return {"version": VERSION, "selection_version": 0,
            "contexts": {}, "entities": {}, "profiles": {}, "taxonomy": {},
            "llm_class": {}, "beliefs": {},
            "researched": [], "llm_asked": [], "selection_words": [],
            "selection_refs": {}, "learning_curve": [],
            "significant_now": False, "capability_confirmed": False}


def _migrate(state: dict) -> dict:
    """v1 -> v2: keep contexts/entities/taxonomy/researched/selection; add the
    belief layer empty.  Learning memory is never discarded (ARCHITECTURE.md 8)."""
    if state.get("version") == VERSION:
        for k, v in _blank().items():
            state.setdefault(k, v)
        return state
    kept = _blank()
    for k in ("contexts", "entities", "taxonomy", "researched",
              "selection_words", "selection_refs", "learning_curve"):
        if k in state:
            kept[k] = state[k]
    return kept


def _observe(state: dict, stories: list[dict]) -> None:
    ctx: dict[str, Counter] = {w: Counter(c) for w, c in state["contexts"].items()}
    ent: Counter = Counter(state.get("entities", {}))
    prof: dict[str, dict] = {w: {"subj": p.get("subj", 0), "obj": p.get("obj", 0),
                                 "subj_verbs": Counter(p.get("subj_verbs", {})),
                                 "obj_verbs": Counter(p.get("obj_verbs", {}))}
                             for w, p in state.get("profiles", {}).items()}
    for story in stories:
        toks: list[str] = []
        for e in story.get("events", []):
            if not isinstance(e, dict) or e.get("provenance") != "heuristic_self":
                continue
            verb = _norm(e.get("verb") or "")
            for slot, pk in (("subject", "subj"), ("obj", "obj")):
                if _is_wordlike(e.get(slot)):
                    w = _norm(e[slot])
                    ent[w] += 1
                    p = prof.setdefault(w, {"subj": 0, "obj": 0,
                                            "subj_verbs": Counter(), "obj_verbs": Counter()})
                    p[pk] += 1
                    if verb:
                        p[f"{pk}_verbs"][verb] += 1
            toks += [_norm(t) for t in (e.get("subject"), e.get("obj"), e.get("verb"))
                     if _is_wordlike(t)]
        for i, w in enumerate(toks):
            near = toks[max(0, i - 4):i] + toks[i + 1:i + 5]
            ctx.setdefault(w, Counter()).update(x for x in near if x != w)
    state["contexts"] = {w: dict(c.most_common(40)) for w, c in ctx.items() if c}
    state["entities"] = dict(ent)
    state["profiles"] = {w: {"subj": p["subj"], "obj": p["obj"],
                             "subj_verbs": dict(Counter(p["subj_verbs"]).most_common(12)),
                             "obj_verbs": dict(Counter(p["obj_verbs"]).most_common(12))}
                         for w, p in prof.items() if p["subj"] or p["obj"]}


def _entity_vocab(state: dict, known_words: "set[str] | None") -> list[str]:
    """Word-like tokens that appeared as a subject/object at least twice -- the
    things worth having a meaning for.  `known_words` (curriculum coverage plus,
    from the reader, the Tatoeba vocab-fuel tokens) restricts the pool so
    research is not spent on one-off parser debris.  Ordered most-read first."""
    ent = state.get("entities", {})
    return [w for w, _ in sorted(ent.items(), key=lambda kv: -kv[1])
            if ent[w] >= 2 and _is_wordlike(w)
            and (known_words is None or w in known_words)]


def _testimony_classes(state: dict, word: str) -> "list[tuple[str, str, float]]":
    """(coarse_class, source, weight) from every testimony source for `word`."""
    out: list[tuple[str, str, float]] = []
    fine = state["taxonomy"].get(word, "")
    c = _coarse(fine)
    if c:
        out.append((c, "wiktionary", 0.22))
    lc = state.get("llm_class", {}).get(word, "")
    if lc in COARSE_CLASSES:
        out.append((lc, "local_model", 0.16))
    return out


def _neighbour_class(state: dict, word: str) -> "tuple[str, float]":
    """A coarse class voted by >=2 co-occurring words that Noise already
    UNDERSTANDS (evidence-backed beliefs only -- the chain grounds out in
    reading, never in a bare testimony vote)."""
    ctx = Counter(state["contexts"].get(word, {}))
    votes = Counter()
    for n, w in ctx.items():
        b = state["beliefs"].get(n)
        if b and b.get("understood") and b.get("genus"):
            votes[b["genus"]] += 1
    if not votes:
        return "", 0.0
    cls, k = votes.most_common(1)[0]
    return (cls, min(0.5, 0.18 * k)) if k >= 2 else ("", 0.0)


def _revise_belief(state: dict, word: str, cycle: int,
                   suppress_testimony: bool = False) -> dict:
    prev = state["beliefs"].get(word, {})
    testimony = [] if suppress_testimony else _testimony_classes(state, word)
    read_cls, read_str = _reading_class(state["profiles"].get(word, {}))
    nbr_cls, nbr_str = _neighbour_class(state, word)

    t_weight: Counter = Counter()
    sources: dict[str, list[str]] = {}
    for cls, src, w in testimony:
        t_weight[cls] += w
        sources.setdefault(cls, []).append(src)
    e_weight: Counter = Counter()
    if read_cls:
        e_weight[read_cls] += read_str
        sources.setdefault(read_cls, []).append("reading_usage")
    if nbr_cls:
        e_weight[nbr_cls] += nbr_str
        sources.setdefault(nbr_cls, []).append("neighbours")

    # a word's OWN usage profile (reading_usage) + testimony is primary evidence;
    # the neighbour vote only corroborates -- "椅子 sits near people" must not
    # outweigh "椅子 is picked up, moved, offered" (a direct observation).
    def _own(cls: str) -> float:
        return min(TESTIMONY_CAP, t_weight.get(cls, 0.0)) + \
            (min(EVIDENCE_HEADROOM, read_str) if read_cls == cls else 0.0)

    classes = set(t_weight) | set(e_weight)
    if not classes:
        belief = {"genus": "", "confidence": 0.0, "understood": False,
                  "support": {"testimony": 0.0, "evidence": 0.0},
                  "sources": [], "revisions": prev.get("revisions", []),
                  "last_cycle": cycle}
        state["beliefs"][word] = belief
        return belief

    best = max(classes, key=lambda c: (_own(c) + (0.15 if nbr_cls == c else 0.0), c == read_cls))
    own_best = _own(best)
    rival = max((_own(c) for c in classes if c != best), default=0.0)
    contest = rival / own_best if own_best > 0 else (1.0 if rival else 0.0)
    nbr_adj = (0.12 if nbr_cls == best else
               -0.1 if (nbr_cls and nbr_cls != best and nbr_str >= 0.35 and own_best < 0.5) else 0.0)
    confidence = round(max(0.0, min(1.0, own_best * (1.0 - 0.45 * contest) + nbr_adj)), 3)
    t_best = min(TESTIMONY_CAP, t_weight.get(best, 0.0))
    e_best = min(EVIDENCE_HEADROOM, e_weight.get(best, 0.0))
    understood = confidence >= UNDERSTOOD_CONF and (
        (read_cls == best and read_str >= UNDERSTOOD_EVIDENCE) or e_best >= UNDERSTOOD_EVIDENCE)

    revisions = list(prev.get("revisions", []))
    if prev.get("genus") and prev["genus"] != best:
        reason = ("reading evidence overruled testimony"
                  if e_weight.get(best, 0.0) > e_weight.get(prev["genus"], 0.0)
                  else "re-weighted")
        revisions.append({"cycle": cycle, "from": prev["genus"], "to": best,
                          "reason": reason})
        revisions = revisions[-12:]

    belief = {"genus": best, "confidence": confidence, "understood": understood,
              "support": {"testimony": round(t_best, 3), "evidence": round(e_best, 3)},
              "sources": sorted(set(sources.get(best, []))),
              "revisions": revisions, "last_cycle": cycle}
    state["beliefs"][word] = belief
    return belief


def _naive_coarse(state: dict, word: str) -> str:
    """Baseline: majority coarse class among the word's co-occurring neighbours'
    RAW dictionary genus -- no confidence, no evidence weighting, no revision."""
    ctx = Counter(state["contexts"].get(word, {}))
    votes = Counter()
    for n in ctx:
        c = _coarse(state["taxonomy"].get(n, ""))
        if c:
            votes[c] += 1
    return votes.most_common(1)[0][0] if votes else ""


def explain(word: str, state: dict, *, allow_self: bool = True) -> dict:
    """Noise's own gloss of `word`.  With allow_self=False the word's own
    testimony (taxonomy entry, local-model class) is excluded from the belief --
    the held-out case: only reading + neighbour beliefs may speak."""
    ctx = Counter(state["contexts"].get(word, {}))
    if allow_self:
        b = state["beliefs"].get(word, {})
        genus, conf, understood = b.get("genus", ""), b.get("confidence", 0.0), b.get("understood", False)
    else:
        read_cls, read_str = _reading_class(state["profiles"].get(word, {}))
        nbr_cls, nbr_str = _neighbour_class(state, word)
        e = Counter()
        if read_cls:
            e[read_cls] += read_str
        if nbr_cls:
            e[nbr_cls] += nbr_str
        genus = max(e, key=e.get) if e else ""
        conf = round(min(EVIDENCE_HEADROOM, e.get(genus, 0.0)), 3) if genus else 0.0
        understood = conf >= UNDERSTOOD_EVIDENCE
    assoc = [w for w, _ in ctx.most_common(EXPLAIN_TERMS)]
    terms = set(assoc) | ({genus} if genus else set())
    if genus:
        tag = "確信" if understood else "推定"
        head = f"「{word}」は{genus}（{tag}{conf:.2f}）"
    else:
        head = f"「{word}」は"
    gloss = head + ("、" + "・".join(assoc) + " に関係する" if assoc else
                    (" まだよく分からない" if not genus else ""))
    return {"word": word, "genus": genus, "confidence": conf, "understood": understood,
            "assoc": assoc, "terms": terms, "gloss": gloss}


def _score_against_ref(expl: dict, ref: dict) -> float:
    """Overlap of the self-explanation's content with the reference definition."""
    ref_terms = set(ref.get("terms", [])) | ({ref["genus"]} if ref.get("genus") else set()) \
        | set(ref.get("related", []))
    if not ref_terms:
        return 0.0
    hit = len(expl["terms"] & ref_terms)
    genus_bonus = 0.5 if (expl["genus"] and ref.get("genus")
                          and (expl["genus"] == ref["genus"]
                               or expl["genus"] in ref["genus"] or ref["genus"] in expl["genus"])) else 0.0
    return min(1.0, hit / min(6, len(ref_terms)) + genus_bonus)


def learn_and_evaluate(stories: list[dict], previous: dict | None, cycle: int,
                       known_words: "set[str] | None" = None,
                       llm: "_LLMClient | None" = None) -> dict:
    state = _migrate(dict(previous or _blank()))
    _observe(state, stories)

    # re-freeze the held-out set / scrub the taxonomy when the policy bumps
    if state.get("selection_version") != SELECTION_VERSION:
        state["selection_version"] = SELECTION_VERSION
        state["selection_words"] = []
        state["selection_refs"] = {}
        state["taxonomy"] = {w: g for w, g in state["taxonomy"].items()
                             if _plausible_genus(g)}

    vocab = _entity_vocab(state, known_words)
    train_words = [w for w in vocab if not _held_out(w)]
    test_words = [w for w in vocab if _held_out(w)]
    held_out_set = set(state["selection_words"])

    # Grow the held-out set lazily and VALIDATED: a candidate joins only if
    # ja.wiktionary gives it a genus that folds onto a concrete coarse class.
    # This keeps the frozen set to words with a real denotation to learn --
    # the literary corpus's frequent "entities" are mostly names / abstractions.
    if len(state["selection_words"]) < TEST_SET_TARGET:
        seen = held_out_set | {w for w in state["selection_refs"]}
        candidates = [w for w in test_words
                      if _is_wordlike(w) and w not in seen and w not in _ABSTRACT_ROLE
                      and not w.endswith(("たち", "さん", "ちゃん", "さま"))][:TEST_PROBES_PER_CYCLE]
        for w in candidates:
            gist = _wiktionary_gist(w) or {}
            state["selection_refs"][w] = gist            # cache probe result (+/-)
            if _coarse(gist.get("genus", "")) in COARSE_CLASSES:
                state["selection_words"].append(w)
        held_out_set = set(state["selection_words"])
    frozen_test = [w for w in state["selection_words"] if w in state["contexts"]]

    # --- testimony: research TRAIN words in ja.wiktionary --------------------
    researched = set(state["researched"])
    pending = [w for w in train_words if w not in researched and w not in held_out_set]
    queue = pending[:1] + sorted(
        pending[1:], key=lambda w: hashlib.sha256(f"rq:{w}".encode()).hexdigest())
    budget = RESEARCH_PER_CYCLE
    for w in queue:
        if budget <= 0:
            break
        gist = _wiktionary_gist(w)
        researched.add(w)
        budget -= 1
        if gist:
            if gist["genus"]:
                state["taxonomy"][w] = gist["genus"]
            for t in (gist["terms"][:5] + gist["related"][:5]):
                if _is_content(t):
                    state["contexts"].setdefault(w, {})
                    state["contexts"][w][t] = state["contexts"][w].get(t, 0) + 2
    state["researched"] = sorted(researched)

    # --- testimony: ask the optional local model a closed-set question ------
    llm = llm if llm is not None else _LLMClient()
    asked = set(state.get("llm_asked", []))
    llm_status = "skipped"
    if llm.available():
        llm_status = "available"
        ask_queue = [w for w in train_words
                     if w not in asked and w not in held_out_set][:LLM_ASKS_PER_CYCLE]
        for w in ask_queue:
            cls = llm.ask_class(w)
            asked.add(w)
            if cls:
                state["llm_class"][w] = cls
    state["llm_asked"] = sorted(asked)

    # --- revise every entity's belief -------------------------------------
    # two passes: neighbour votes read last cycle's beliefs, so settle once more.
    # A held-out word's stored belief is evidence-only -- its testimony is
    # suppressed so it can neither be scored on it nor pass it to a neighbour.
    for _ in range(2):
        for w in vocab:
            _revise_belief(state, w, cycle, suppress_testimony=w in held_out_set)
    understood_now = sum(1 for w in vocab if state["beliefs"].get(w, {}).get("understood"))
    corrections = sum(len(state["beliefs"].get(w, {}).get("revisions", [])) for w in vocab)

    # --- capability: classify the frozen held-out words -------------------
    per_gain: list[float] = []
    belief_hits = 0
    measured = 0
    for w in frozen_test[:20]:
        ref = state["selection_refs"].get(w) or {}
        ref_coarse = _coarse(ref.get("genus", ""))
        if ref_coarse not in COARSE_CLASSES:
            continue
        # belief classification with testimony on this word suppressed
        held_expl = explain(w, state, allow_self=False)
        belief_correct = 1.0 if (held_expl["understood"]
                                 and _compatible(held_expl["genus"], ref_coarse)) else 0.0
        base_correct = 1.0 if _compatible(_naive_coarse(state, w), ref_coarse) else 0.0
        per_gain.append(belief_correct - base_correct)
        belief_hits += belief_correct
        measured += 1

    n = len(per_gain)
    mean_gain = sum(per_gain) / n if n else 0.0
    understood_rate = round(belief_hits / n, 3) if n else 0.0
    if n >= 2:
        var = sum((g - mean_gain) ** 2 for g in per_gain) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        z = mean_gain / se if se > 0 else (99.0 if mean_gain > 0 else 0.0)
    else:
        z = 0.0
    significant = n >= MIN_TEST_WORDS and z >= SIGNIFICANCE_Z and mean_gain > 0
    prior_sig = previous.get("significant_now", False) if previous else False

    curve = list(state.get("learning_curve", []))
    point = {"cycle": cycle, "vocab": len(vocab), "researched": len(state["researched"]),
             "beliefs": len(state["beliefs"]), "understood": understood_now,
             "corrections": corrections, "test_words": len(frozen_test),
             "understood_rate": understood_rate,
             "mean_gain": round(mean_gain, 4), "z": round(z, 2)}
    if not curve or curve[-1]["cycle"] != cycle:
        curve.append(point)
    state["learning_curve"] = curve[-200:]
    state["significant_now"] = significant
    state["capability_confirmed"] = bool(significant and prior_sig)

    return {**state, "status": "measured" if n else "insufficient_test_words",
            "vocab": len(vocab), "researched_count": len(state["researched"]),
            "taxonomy_size": len(state["taxonomy"]),
            "belief_count": len(state["beliefs"]), "understood_count": understood_now,
            "understood_rate": understood_rate, "corrections": corrections,
            "llm_status": llm_status, "llm_asked_count": len(state["llm_asked"]),
            "test_words": len(frozen_test), "measured": measured,
            "mean_gain": round(mean_gain, 4), "z": round(z, 2),
            "significant_now": significant,
            "capability_confirmed": state["capability_confirmed"],
            "sample_explanations": [explain(w, state, allow_self=False)["gloss"]
                                    for w in frozen_test[:3]],
            "license_note": "meaning learned from ja.wiktionary / ja.wikipedia "
                            "(CC-BY-SA) and an optional local model; all testimony "
                            "enters a belief at a capped weight and is never stored "
                            "verbatim -- a word is 'understood' only on Noise's own "
                            "reading evidence"}
