#!/usr/bin/env python3
"""Learn what Japanese words MEAN, tested by self-explanation.

The reading loop's "comprehension" is structural: it never represents what a
word denotes.  This module closes that gap without an LLM:

  * distributional semantics -- for every content word Noise reads (its own
    heuristic (subject, obj, verb) tokens), accumulate the content words it
    co-occurs with.  Words in similar contexts have similar meaning.
  * grounded genus -- for a bounded number of TRAIN words per cycle, fetch the
    ja.wiktionary definition (CC-BY-SA) and extract the hypernym ("...イヌ科に
    属する哺乳動物" -> 哺乳動物) plus the definition's key terms.  A word -> genus
    taxonomy accumulates.
  * self-explanation -- `explain(word)` produces Noise's own gloss ("「きつね」は
    動物で、稲荷・耳・化かす に関係する") from the co-occurrence memory and the
    taxonomy (propagating a genus through near neighbours), NEVER from the word's
    own stored definition.

Capability (frozen, held-out): a fixed set of words is NEVER researched.  Noise
must explain them from reading + the taxonomy learned on OTHER words.  Score =
overlap of the self-explanation's content with the reference ja.wiktionary
definition, vs a no-taxonomy frequency baseline, one-sided significance.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from collections import Counter

from web_cache import WEB_CACHE, NetworkBudgetExceeded

VERSION = 1
USER_AGENT = "AI_Noise/0.31 (developmental Japanese word meaning; read-only)"
WIKTIONARY_API = "https://ja.wiktionary.org/w/api.php"
WIKIPEDIA_API = "https://ja.wikipedia.org/w/api.php"
HELD_OUT_SALT = "japanese-word-meaning:heldout:v1"
HELD_OUT_FRACTION = 5          # ~1/5 of the vocabulary is the frozen test set
RESEARCH_PER_CYCLE = 3         # ja.wiktionary fetches per cycle (train words only)
MIN_TEST_WORDS = 8
SIGNIFICANCE_Z = 3.0
EXPLAIN_TERMS = 5

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


def _held_out(word: str) -> bool:
    h = hashlib.sha256(f"{HELD_OUT_SALT}:{word}".encode()).hexdigest()
    return int(h[:8], 16) % HELD_OUT_FRACTION == 0


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[ぁ-ゟ゠-ヿ一-鿿]{2,}", text or "") if _is_content(w)]


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
        if cand not in _STOP and cand not in _BAD_GENUS and word not in cand and not cand.startswith("で"):
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
        if c not in _BAD_GENUS and c not in _STOP and word not in c:
            return c
    return ""


# --- learning + explanation ---------------------------------------------
def _blank() -> dict:
    return {"version": VERSION, "contexts": {}, "entities": {}, "taxonomy": {},
            "researched": [], "selection_words": [], "selection_refs": {},
            "learning_curve": [], "capability_confirmed": False}


def _observe(state: dict, stories: list[dict]) -> None:
    ctx: dict[str, Counter] = {w: Counter(c) for w, c in state["contexts"].items()}
    ent: Counter = Counter(state.get("entities", {}))
    for story in stories:
        toks: list[str] = []
        for e in story.get("events", []):
            if not isinstance(e, dict) or e.get("provenance") != "heuristic_self":
                continue
            for slot in ("subject", "obj"):
                if _is_content(e.get(slot)):
                    ent[_norm(e[slot])] += 1
            toks += [_norm(t) for t in (e.get("subject"), e.get("obj"), e.get("verb"))
                     if _is_content(t)]
        for i, w in enumerate(toks):
            near = toks[max(0, i - 4):i] + toks[i + 1:i + 5]
            ctx.setdefault(w, Counter()).update(x for x in near if x != w)
    state["contexts"] = {w: dict(c.most_common(40)) for w, c in ctx.items() if c}
    state["entities"] = dict(ent)


def _entity_vocab(state: dict, known_words: "set[str] | None") -> list[str]:
    """Content words that appeared as a subject/object at least twice -- the
    things worth having a meaning for.  Ordered most-read first."""
    ent = state.get("entities", {})
    return [w for w, _ in sorted(ent.items(), key=lambda kv: -kv[1])
            if ent[w] >= 2 and _is_content(w)
            and (known_words is None or w in known_words)]


def explain(word: str, state: dict, *, allow_self: bool = True) -> dict:
    """Noise's own gloss of `word` -- genus + associations, from the
    co-occurrence memory and the taxonomy.  With allow_self=False the word's own
    research (taxonomy entry, related terms) is excluded -- the held-out case."""
    ctx = Counter(state["contexts"].get(word, {}))
    tax = state["taxonomy"]
    genus = tax.get(word, "") if allow_self else ""
    if not genus:
        # propagate: the most common genus among this word's context neighbours
        votes = Counter(tax[n] for n in ctx if n in tax and (allow_self or n != word))
        if votes:
            genus = votes.most_common(1)[0][0]
    assoc = [w for w, _ in ctx.most_common(EXPLAIN_TERMS)]
    terms = set(assoc) | ({genus} if genus else set())
    gloss = (f"「{word}」は{genus}で、" if genus else f"「{word}」は") + \
            ("・".join(assoc) + " に関係する" if assoc else "まだよく分からない")
    return {"word": word, "genus": genus, "assoc": assoc, "terms": terms, "gloss": gloss}


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
                       known_words: "set[str] | None" = None) -> dict:
    state = dict(previous or _blank())
    if state.get("version") != VERSION:
        state = _blank()
    for k, v in _blank().items():
        state.setdefault(k, v)
    _observe(state, stories)

    vocab = _entity_vocab(state, known_words)
    train_words = [w for w in vocab if not _held_out(w)]
    test_words = [w for w in vocab if _held_out(w)]

    # freeze the selection (held-out) set once
    if not state["selection_words"] and len(test_words) >= MIN_TEST_WORDS:
        state["selection_words"] = test_words[:40]
    frozen_test = [w for w in state["selection_words"] if w in state["contexts"]]

    # research TRAIN entity words: one most-read, the rest sampled across the
    # frequency range (common nouns like きつね/はな are rarely the top entity in
    # a literary corpus but define well)
    researched = set(state["researched"])
    pending = [w for w in train_words if w not in researched]
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

    # capability: explain the frozen held-out words from reading + taxonomy only,
    # scored against their reference definitions (fetched for scoring, cached)
    per_gain: list[float] = []
    measured = 0
    for w in frozen_test[:20]:
        ref = state["selection_refs"].get(w)
        if ref is None:
            ref = _wiktionary_gist(w) or {}
            state["selection_refs"][w] = ref
        if not ref:
            continue
        learned = _score_against_ref(explain(w, state, allow_self=False), ref)
        base_expl = {"genus": "", "assoc": [w2 for w2, _ in
                     Counter(state["contexts"].get(w, {})).most_common(EXPLAIN_TERMS)]}
        base_expl["terms"] = set(base_expl["assoc"])
        base = _score_against_ref(base_expl, ref)
        per_gain.append(learned - base)
        measured += 1

    n = len(per_gain)
    mean_gain = sum(per_gain) / n if n else 0.0
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
             "taxonomy": len(state["taxonomy"]), "test_words": len(frozen_test),
             "mean_gain": round(mean_gain, 4), "z": round(z, 2)}
    if not curve or curve[-1]["cycle"] != cycle:
        curve.append(point)
    state["learning_curve"] = curve[-200:]
    state["significant_now"] = significant
    state["capability_confirmed"] = bool(significant and prior_sig)

    return {**state, "status": "measured" if n else "insufficient_test_words",
            "vocab": len(vocab), "researched_count": len(state["researched"]),
            "taxonomy_size": len(state["taxonomy"]),
            "test_words": len(frozen_test), "measured": measured,
            "mean_gain": round(mean_gain, 4), "z": round(z, 2),
            "significant_now": significant,
            "capability_confirmed": state["capability_confirmed"],
            "sample_explanations": [explain(w, state, allow_self=False)["gloss"]
                                    for w in frozen_test[:3]],
            "license_note": "meaning learned from ja.wiktionary (CC-BY-SA); "
                            "definitions are not stored or reproduced"}
