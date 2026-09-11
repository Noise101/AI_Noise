#!/usr/bin/env python3
"""Extract simple Japanese properties, classes, possession and locations.

The action-event parser intentionally represents changes and actions.  Ordinary
Japanese also teaches meaning through stative clauses, which must not be forced
into a fake verb slot:

  猫は動物です。       -> 猫 | is | 動物
  レモンは黄色い。     -> レモン | is | 黄色い
  犬には足がある。     -> 犬 | has | 足
  本は机の上にある。   -> 本 | at | 上

These are revisable observations from text, not dictionary truth.  Extraction
uses a disclosed morphological tokenizer when available and fails closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import morphology_teacher as morphology


VERSION = 1
PROVENANCE = "proposition_self"
_SPLIT = re.compile(r"(?<=[。！？])|\n+")
_PUNCT = "。！？、,.「」『』（）() \t\n"
_NON_ENTITY = {"これ", "それ", "あれ", "もの", "こと", "ところ", "とき", "ため",
               "よう", "そう", "何", "なに", "ここ", "そこ", "どこ", "の", "とこ"}
_NON_VALUE = _NON_ENTITY | {"誰", "だれ", "何者", "ない"}
_NON_SUBJECT = _NON_ENTITY | {"時", "方", "中", "上", "下", "前", "後", "間"}
_FALLBACK_TOPIC = re.compile(
    r"^(?P<subject>[ァ-ヿ一-鿿々ー]{1,12}?)(?:と|に)?は(?P<body>.+)$")


@dataclass(frozen=True)
class JapaneseProposition:
    subject: str
    relation: str
    value: str
    polarity: str
    sentence: str
    provenance: str = PROVENANCE

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.value}"


def _clean(value: str) -> str:
    return re.sub(r"[^ぁ-ゟ゠-ヿ一-鿿々ー]", "", value or "")


def _valid_noun(value: str) -> bool:
    return bool(value and value not in _NON_ENTITY and 1 <= len(value) <= 12)


def _noun_before(ms: list, index: int) -> str:
    parts = []
    i = index - 1
    while i >= 0:
        m = ms[i]
        if m.pos in ("名詞", "代名詞"):
            parts.append(m.surface)
            i -= 1
            continue
        if m.pos == "接頭詞":
            i -= 1
            continue
        break
    return _clean("".join(reversed(parts[-3:])))


def _noun_span(ms: list, start: int, end: int) -> str:
    nouns = [m.surface for m in ms[start:end] if m.pos in ("名詞", "代名詞")]
    return _clean("".join(nouns[-2:]))


def _negative(ms: list) -> bool:
    return any((m.base or m.surface) in ("ない", "ぬ") or m.surface in ("ん", "ません")
               for m in ms)


def _terminal_predicate(ms: list, index: int) -> bool:
    """True only when no second content predicate follows ``index``."""
    return all(m.pos in ("助動詞", "記号") for m in ms[index + 1:])


def _only_pos(ms: list, allowed: set[str]) -> bool:
    return all(m.pos in allowed for m in ms)


def _fallback(text: str, sentence: str) -> JapaneseProposition | None:
    """Conservative analyser-free path for the most explicit constructions.

    This intentionally has low coverage.  It requires overt は/には and an
    overt copula, existence verb, or common adjective inflection; ambiguous
    free-form predicates are left unknown instead of becoming fake facts.
    """
    match = _FALLBACK_TOPIC.match(text)
    if not match:
        return None
    subject, body = match.group("subject"), match.group("body")
    if not _valid_noun(subject) or subject in _NON_SUBJECT:
        return None

    # Xには（数量・の）Yがある/いる
    if "には" in text:
        possession = re.fullmatch(
            r"(?:[一-鿿〇零一二三四五六七八九十百千万0-9]+(?:本|個|匹|枚|つ)?の?)?"
            r"(?P<item>[ァ-ヿ一-鿿々ー]{1,12})が(?P<neg>ない|いない|ありません|いません|ある|いる)",
            body)
        if possession:
            item = possession.group("item")
            neg = possession.group("neg") in ("ない", "いない", "ありません", "いません")
            return JapaneseProposition(subject, "has-not" if neg else "has", item,
                                       "negative" if neg else "positive", sentence)

    # XはYにある/いる and XはYである/いる.  The copular である case is
    # handled below first so it is not mistaken for location.
    nominal = re.fullmatch(
        r"(?P<value>[ァ-ヿ一-鿿々ー]{1,12})(?P<copula>です|だ|である|ではない|じゃない)", body)
    if nominal:
        value = nominal.group("value")
        neg = nominal.group("copula") in ("ではない", "じゃない")
        if _valid_noun(value) and value not in _NON_VALUE and value != subject:
            return JapaneseProposition(subject, "is-not" if neg else "is", value,
                                       "negative" if neg else "positive", sentence)

    location = re.fullmatch(
        r"(?P<place>[ぁ-ゟァ-ヿ一-鿿々ー]{1,18})(?:に|で)(?P<neg>ない|いない|ある|いる)", body)
    if location:
        place = location.group("place").split("の")[-1]
        neg = location.group("neg") in ("ない", "いない")
        if _valid_noun(place):
            return JapaneseProposition(subject, "not-at" if neg else "at", place,
                                       "negative" if neg else "positive", sentence)

    # Dictionary-form and the two common past/negative inflections.  Requiring
    # a kanji-bearing stem keeps all-kana verb wishes such as したい out.
    adjective = re.fullmatch(
        r"(?P<stem>[一-鿿々][ぁ-ゟ一-鿿々ー]{0,10}?)(?P<ending>くなかった|くない|かった|い)", body)
    if adjective:
        stem, ending = adjective.group("stem"), adjective.group("ending")
        value = stem + "い"
        neg = ending in ("くない", "くなかった")
        if value not in _NON_VALUE:
            return JapaneseProposition(subject, "is-not" if neg else "is", value,
                                       "negative" if neg else "positive", sentence)
    return None


def extract_proposition(sentence: str) -> JapaneseProposition | None:
    if (sentence or "").rstrip().endswith(("？", "?")):
        return None
    text = (sentence or "").strip(_PUNCT)
    if not 3 <= len(text) <= 40:
        return None
    # A comma usually joins multiple clauses.  Attaching a later predicate to
    # the first は-topic created plausible-looking false facts in literary text.
    # v1 deliberately prefers low recall to polluted semantic memory.
    if "、" in text or "," in text:
        return None
    analysis = morphology.analyse(text)
    if not analysis or not analysis.morphemes:
        return _fallback(text, sentence)
    ms = analysis.morphemes

    # XにはYがある/いる -> X has Y.  Require both overt markers; ordinary
    # XがYにある is location, not possession.
    for i in range(1, len(ms) - 3):
        if ms[i].surface != "に" or ms[i + 1].surface != "は":
            continue
        owner = _noun_before(ms, i)
        if _clean("".join(m.surface for m in ms[:i])) != owner:
            continue
        ga = next((j for j in range(i + 2, len(ms)) if ms[j].surface == "が"), None)
        if ga is None:
            continue
        item = _noun_before(ms, ga)
        existence_i = next((j for j in range(ga + 1, len(ms))
                            if (ms[j].base or ms[j].surface) in ("ある", "いる")), None)
        if (existence_i is not None and _terminal_predicate(ms, existence_i)
                and not ms[ga + 1:existence_i]
                and _valid_noun(owner) and _valid_noun(item)):
            neg = _negative(ms[ga + 1:])
            return JapaneseProposition(owner, "has-not" if neg else "has", item,
                                       "negative" if neg else "positive", sentence)

    # Topic marker.  「XとはY」 is tokenized as と + は, so skip the と.
    topic = next((i for i, m in enumerate(ms) if m.surface == "は"), None)
    if topic is None or topic > 4:
        return None
    subject_marker = topic - 1 if topic > 0 and ms[topic - 1].surface == "と" else topic
    subject = _noun_before(ms, subject_marker)
    if (not _valid_noun(subject) or subject in _NON_SUBJECT
            or _clean("".join(m.surface for m in ms[:subject_marker])) != subject):
        return None
    start = topic + 1
    neg = _negative(ms[start:])

    # XはYではない.  This has a second は, so recognise the tightly bounded
    # noun-only negative copula before rejecting other multiple-topic clauses.
    if sum(1 for m in ms if m.surface == "は") == 2:
        de = next((i for i in range(start, len(ms) - 1) if ms[i].surface == "で"), None)
        second_wa = de + 1 if de is not None else None
        if (de is not None and second_wa < len(ms) and ms[second_wa].surface == "は"
                and _only_pos(ms[start:de], {"名詞", "接頭詞"})
                and any((m.base or m.surface) == "ない" for m in ms[second_wa + 1:])):
            value = _noun_span(ms, start, de)
            if _valid_noun(value) and value != subject:
                return JapaneseProposition(subject, "is-not", value, "negative", sentence)
        return None
    if sum(1 for m in ms if m.surface == "は") != 1:
        return None

    # XはYにある/いる -> X at Y.
    loc = next((i for i in range(start + 1, len(ms)) if ms[i].surface == "に"), None)
    existence_i = (next((i for i in range(loc + 1, len(ms))
                         if (ms[i].base or ms[i].surface) in ("ある", "いる")), None)
                   if loc is not None else None)
    if (existence_i is not None and _terminal_predicate(ms, existence_i)
            and not ms[loc + 1:existence_i]):
        place = _noun_before(ms, loc)
        if _valid_noun(place):
            return JapaneseProposition(subject, "not-at" if neg else "at", place,
                                       "negative" if neg else "positive", sentence)

    # Xは黄色い / Xは悲しかった.  Janome gives the dictionary adjective in base.
    adjective = next((m for m in ms[start:] if m.pos == "形容詞"), None)
    if (adjective and _terminal_predicate(ms, ms.index(adjective))
            and not any(m.pos == "動詞" for m in ms[start:])):
        value = _clean(adjective.base or adjective.surface)
        if value and value not in _NON_VALUE:
            return JapaneseProposition(subject, "is-not" if neg else "is", value,
                                       "negative" if neg else "positive", sentence)

    # XはYです/だ/である.  A copula is an auxiliary in IPADIC; the complement
    # immediately before it must contain a noun.  This deliberately excludes
    # action predicates and passive/progressive clauses.
    copula = next((i for i in range(start, len(ms))
                   if ms[i].pos == "助動詞"
                   and (ms[i].base or ms[i].surface) in ("だ", "です")), None)
    if (copula is not None and _terminal_predicate(ms, copula)
            and _only_pos(ms[start:copula], {"名詞", "接頭詞"})):
        value = _noun_span(ms, start, copula)
        if _valid_noun(value) and value not in _NON_VALUE and value != subject:
            return JapaneseProposition(subject, "is-not" if neg else "is", value,
                                       "negative" if neg else "positive", sentence)
    return None


def extract_story(text: str) -> list[JapaneseProposition]:
    out = []
    for raw in _SPLIT.split(text or ""):
        prop = extract_proposition(raw)
        if prop:
            out.append(prop)
    return out
