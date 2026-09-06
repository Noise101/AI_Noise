#!/usr/bin/env python3
"""Particle-anchored event extraction for Japanese children's stories.

The English parser (narrative_event_v29) has to guess grammatical roles from
word order and drops ~56% of sentences.  Japanese marks roles explicitly with
case particles, so a much simpler and more reliable extractor is possible:

  きつねが ぶどうを 見つけた  ->  きつね | 見つける | ぶどう
      が (subject)  を (object)   verb (normalised to dictionary form)

Design choices, all rule-based, no morphological analyser, no pretrained model:
  * full word segmentation is NOT attempted -- particles are the only anchors we
    need.  An argument is the run of Japanese characters immediately before a
    case particle, back to the previous particle / punctuation / clause start.
  * zero anaphora (Japanese omits the subject constantly) is resolved to the
    most recent topic/subject, threaded across the story.
  * verb normalisation covers the polite forms (ます/ました/ません) that graded
    readers and folktale retellings overwhelmingly use, plus a table of the
    most common irregular and godan verbs.  Anything else is kept as-is and the
    event is marked low-confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# character classes
HIRAGANA = r"ぁ-ゖゝゞ"
KATAKANA = r"ァ-ヺー"
KANJI = r"㐀-䶿一-鿿"
JP = f"[{HIRAGANA}{KATAKANA}{KANJI}]"
JP_RUN = re.compile(f"{JP}+")
RUBY = re.compile(r"《[^》]*》|｜")            # aozora-style ruby markers -> strip
SENT_SPLIT = re.compile(r"(?<=[。！？])")

# Case particles, longest first.  が/を are the reliable anchors; に/へ/から
# introduce obliques; は/も are the topic and are only matched at clause start
# (mid-run they are usually word-internal: もも, おも..., ...も).
STRONG_PARTICLES = ["から", "が", "を", "に", "へ", "で", "と"]
TOPIC_PARTICLES = ["は", "も"]
SUBJECT_MARKERS = ("が", "は", "も")
OBJECT_MARKERS = ("を",)

SENTENCE_END = "。！？、」』）\n 　"
CONNECTIVES = {"そして", "それから", "すると", "しかし", "けれども", "でも",
               "ところが", "やがて", "また", "その", "この", "あの", "ある",
               "むかし", "むかしむかし", "とても", "やがて", "ある日", "いつも"}
# common nouns that embed a particle character -- protect them from the scanner
PROTECTED_NOUN_HEADS = ("もも", "おに", "かに", "とり", "にわ", "には虫", "きのこ",
                        "はな", "はた", "はし", "はこ", "もり", "こども", "ともだち",
                        "でんち", "にもつ", "におい")
# leading adjectival / determiner / adverbial prefixes to strip off a noun
NOUN_PREFIX = re.compile(r"^(大きな|小さな|きれいな|りっぱな|元気な|かわいい|やさしい|"
                         r"わるい|いい|ある|その|この|あの|ひとつの|一つの|"
                         r"まだ|もう|ずっと|やがて|すぐ|とても|いつも|きっと|"
                         r"でも|そして|それから|すると|しかし|ところが|また)")
# 「... という ...」: the noun before という names the entity that follows
TO_IU = re.compile(r"という")

# common verbs: any inflected surface -> dictionary form.  Kept small and
# high-frequency; the ます/ました/た rules below handle the long tail.
VERB_TABLE = {
    "する": "する", "した": "する", "して": "する", "します": "する", "しました": "する",
    "しない": "する", "せず": "する", "できた": "できる", "できる": "できる",
    "来た": "来る", "きた": "来る", "来る": "来る", "くる": "来る", "来ました": "来る", "きました": "来る",
    "行った": "行く", "いった": "行く", "行く": "行く", "いく": "行く", "行きました": "行く",
    "あった": "ある", "ある": "ある", "あります": "ある", "ありました": "ある", "ない": "ある",
    "いた": "いる", "いる": "いる", "います": "いる", "いました": "いる",
    "なった": "なる", "なる": "なる", "なりました": "なる", "なります": "なる",
    "言った": "言う", "いった": "言う", "言いました": "言う", "いいました": "言う", "言う": "言う",
    "見た": "見る", "みた": "見る", "見ました": "見る", "みました": "見る", "見る": "見る", "見つけた": "見つける",
    "食べた": "食べる", "たべた": "食べる", "食べました": "食べる", "食べる": "食べる",
    "取った": "取る", "とった": "取る", "取りました": "取る",
    "作った": "作る", "つくった": "作る", "作りました": "作る",
    "帰った": "帰る", "かえった": "帰る", "帰りました": "帰る",
    "出た": "出る", "でた": "出る", "出ました": "出る", "入った": "入る", "はいった": "入る",
    "生まれた": "生まれる", "うまれた": "生まれる", "生まれました": "生まれる",
    "くれた": "くれる", "くれました": "くれる", "もらった": "もらう", "あげた": "あげる",
}
# godan: i-row (ます-stem last kana) -> dictionary u-row
I_TO_U = {"い": "う", "き": "く", "ぎ": "ぐ", "し": "す", "ち": "つ",
          "に": "ぬ", "ひ": "ふ", "び": "ぶ", "み": "む", "り": "る"}
# past-tense godan endings: surface tail -> (drop, dictionary tail)
GODAN_PAST = {"った": ["う", "つ", "る"], "いた": ["く"], "いだ": ["ぐ"],
              "した": ["す"], "んだ": ["ぬ", "ぶ", "む"]}


def normalise_text(text: str) -> str:
    return RUBY.sub("", text).replace("　", " ")


TE_AUX = re.compile(r"[てで](き(た|ました|ます)|くる|きます|"
                    r"い(た|ました|ます|きました|る)|いく|"
                    r"しま(った|いました|う)|お(いた|きました)|み(た|ました|る))$")


def _dictionary_verb(surface: str) -> tuple[str, float]:
    """(dictionary form, confidence)."""
    surface = surface.strip("。、！？「」『』（）　 \n")
    if not surface:
        return "", 0.0
    # strip a subsidiary て-verb (流れてきた -> 流れる, 持っていった -> 持つ)
    aux = TE_AUX.search(surface)
    if aux and aux.start() > 1:
        surface = surface[:aux.start() + 1]     # keep the main te-form (incl. て/っ)
        if surface.endswith("って"):
            surface = surface[:-2] + "った"      # 持って -> 持った for the rules below
        elif surface.endswith("んで"):
            surface = surface[:-2] + "んだ"      # 住んで -> 住んだ
        elif surface.endswith("いで"):
            surface = surface[:-2] + "いだ"      # 泳いで -> 泳いだ
        elif surface.endswith("て"):
            surface = surface[:-1] + "た"        # 流れて -> 流れた
    if surface in VERB_TABLE:
        return VERB_TABLE[surface], 0.95
    # polite: ...ます / ...ました / ...ません(でした) / ...まして / ...ましょう
    m = re.search(r"(まし(た|て)|ます|ませんでした|ません|ましょう)$", surface)
    if m:
        stem = surface[:m.start()]
        if not stem:
            return surface, 0.3
        last = stem[-1]
        if last in I_TO_U:                       # godan: 買い -> 買う
            return stem[:-1] + I_TO_U[last], 0.8
        return stem + "る", 0.75                 # ichidan: 食べ -> 食べる
    # plain past: ...た / ...だ
    if surface.endswith(("た", "だ")):
        for tail, options in GODAN_PAST.items():
            if surface.endswith(tail):
                return surface[:-len(tail)] + options[0], 0.6
        if surface.endswith("た"):               # ichidan past: 食べた -> 食べる
            return surface[:-1] + "る", 0.65
    # negative: ...ない
    if surface.endswith("ない") and len(surface) > 2:
        stem = surface[:-2]
        a_to_u = {"わ": "う", "か": "く", "が": "ぐ", "さ": "す", "た": "つ",
                  "な": "ぬ", "ば": "ぶ", "ま": "む", "ら": "る"}
        if stem and stem[-1] in a_to_u:
            return stem[:-1] + a_to_u[stem[-1]], 0.55
        return stem + "る", 0.5
    if surface.endswith(("う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む", "る")):
        return surface, 0.7                      # already dictionary-ish
    return surface, 0.2


@dataclass(frozen=True)
class JapaneseEvent:
    subject: str
    verb: str
    obj: str
    confidence: float
    sentence: str
    roles: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.verb}|{self.obj}"


_KANJI_KATA = re.compile(f"[{KATAKANA}{KANJI}]")


def _clean_noun(raw: str) -> str:
    noun = re.sub(f"[^{HIRAGANA}{KATAKANA}{KANJI}]", "", raw)
    noun = NOUN_PREFIX.sub("", noun)
    return noun


def _noun_ok(raw: str) -> bool:
    """A plausible NP head: >=2 chars, OR a single kanji/katakana (山, 川, 犬)."""
    noun = _clean_noun(raw)
    return len(noun) >= 2 or (len(noun) == 1 and bool(_KANJI_KATA.match(noun)))


def _split_particles(clause: str, known_words: "set[str] | None" = None) -> list[tuple[str, str]]:
    """[(noun, particle), ...] for the case-marked NPs in one clause, in order.

    A particle is only accepted when the run before it is >= 2 characters and
    not a known connective; topic particles (は/も) are accepted only for the
    first NP of the clause.  Substrings in PROTECTED_NOUN_HEADS, and any span
    matching a `known_words` entry, are skipped so the peach in もも is not read
    as the particle も and 上がる is not split at が."""
    out = []
    i = 0
    current = ""
    seen_topic = False
    verb_start = 0
    while i < len(clause):
        for protected in PROTECTED_NOUN_HEADS:
            if clause.startswith(protected, i):
                current += protected
                i += len(protected)
                break
        else:
            matched = None
            nxt = clause[i + 1] if i + 1 < len(clause) else ""
            prev = current[-1] if current else ""
            for particle in STRONG_PARTICLES:
                if not clause.startswith(particle, i) or not _noun_ok(current):
                    continue
                # が followed by an inflection kana is verb-internal (上がる, 転がる)
                if particle == "が" and nxt in "るりっられろ":
                    continue
                # で in でした/です/でしょう or after ん is copula/verb-internal
                if particle == "で" and (nxt in "しす" or prev == "ん"):
                    continue
                # a known word spanning the particle -> not a boundary
                if known_words and any(w for w in (current[-2:] + particle + nxt,
                                                   current[-1:] + particle + nxt)
                                       if w in known_words):
                    continue
                matched = particle
                break
            if matched is None and not seen_topic and not out:
                run = len(_clean_noun(current))
                # topic は only at the first NP after a >=2-char run; topic も
                # needs >=3 (も is far more often word-internal: もも, くも, ...)
                for particle, need in (("は", 2), ("も", 3)):
                    if clause.startswith(particle, i) and run >= need:
                        matched, seen_topic = particle, True
                        break
            if matched:
                noun = _clean_noun(current)
                if noun and noun not in CONNECTIVES:
                    out.append((noun, matched))
                current = ""
                i += len(matched)
                verb_start = i
            else:
                current += clause[i]
                i += 1
    return out, verb_start


def extract_clause(sentence: str, recent_subject: str | None = None,
                   known_words: "set[str] | None" = None) -> JapaneseEvent | None:
    text = normalise_text(sentence).strip()
    if len(text) < 4:
        return None
    core = text.rstrip(SENTENCE_END)
    # 「Xというもの」 -> the entity is X; drop the という so the scanner sees Xが/を
    core = TO_IU.sub("", core, count=1) if TO_IU.search(core) else core
    pairs, verb_start = _split_particles(core, known_words)
    # the verb complex is whatever follows the last matched (noun, particle) --
    # taken straight from the scanner so it agrees with which が/を it accepted
    verb_surface = re.sub(r"^[、。「」『』（）\s]+", "", core[verb_start:])
    verb_run = JP_RUN.search(verb_surface)
    if not verb_run:
        return None
    verb, verb_conf = _dictionary_verb(verb_run.group(0))
    if not verb:
        return None

    subject = obj = ""
    roles: dict[str, str] = {}
    for noun, particle in pairs:
        if particle in SUBJECT_MARKERS and not subject:
            subject = noun
        elif particle in OBJECT_MARKERS and not obj:
            obj = noun
        else:
            roles.setdefault(particle, noun)
    if not subject:
        subject = recent_subject or ""
    if not subject:
        return None
    confidence = round(min(verb_conf, 0.9 if obj or roles else 0.7), 3)
    return JapaneseEvent(subject=subject, verb=verb, obj=obj, confidence=confidence,
                         sentence=sentence.strip(), roles=roles)


def learn_word_vocabulary(text: str, minimum_count: int = 2, top: int = 400) -> set[str]:
    """Unsupervised word candidates from the corpus itself (boundary induction
    by branching-entropy), used to keep the particle scanner out of the middle
    of words.  Mirrors japanese_boundaries_v18's BoundaryInducer, kept local so
    this module stands alone."""
    from collections import Counter, defaultdict
    counts: Counter = Counter()
    left: dict = defaultdict(set)
    right: dict = defaultdict(set)
    for run in JP_RUN.findall(normalise_text(text)):
        for size in (2, 3, 4, 5, 6):
            for index in range(len(run) - size + 1):
                chunk = run[index:index + size]
                counts[chunk] += 1
                left[chunk].add(run[index - 1] if index else "^")
                right[chunk].add(run[index + size] if index + size < len(run) else "$")
    import math
    scored = []
    for form, count in counts.items():
        if count < minimum_count:
            continue
        variety = len(left[form]) + len(right[form])
        scored.append((count * (len(form) - 1) * (1 + math.log2(max(1, variety))), form))
    return {form for _, form in sorted(scored, reverse=True)[:top]}


def extract_story(text: str, known_words: "set[str] | None" = None) -> list[JapaneseEvent]:
    """Ordered events for a whole story, threading the omitted subject."""
    vocab = known_words if known_words is not None else learn_word_vocabulary(text)
    events: list[JapaneseEvent] = []
    recent_subject: str | None = None
    for raw in SENT_SPLIT.split(normalise_text(text)):
        sentence = raw.strip()
        if not sentence:
            continue
        for clause in re.split(r"、(?=\S)", sentence):
            event = extract_clause(clause if clause.endswith(tuple("。！？")) else clause + "。",
                                   recent_subject, vocab)
            if event and event.confidence >= 0.4:
                events.append(event)
                recent_subject = event.subject
    return events
