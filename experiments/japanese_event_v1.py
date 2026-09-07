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

# Bumped when extract_story changes enough that a book set aside as unparsable /
# stuck under the old behaviour deserves a fresh reading (reading_curriculum_v1
# stamps this per book; japanese_reader_v1 re-shelves the stragglers).
#   1 -> initial per-fragment extractor
#   2 -> sentence-level clause chaining + relative-clause / predicate-が fixes
PARSER_VERSION = 2

# character classes
HIRAGANA = r"ぁ-ゖゝゞ"
KATAKANA = r"ァ-ヺー"
KANJI = r"㐀-䶿一-鿿"
JP = f"[{HIRAGANA}{KATAKANA}{KANJI}]"
JP_RUN = re.compile(f"{JP}+")
RUBY = re.compile(r"《[^》]*》|｜")            # aozora-style ruby markers -> strip
SENT_SPLIT = re.compile(r"(?<=[。！？])")

# 「…」 direct speech: the quote content is pulled out before sentence splitting
# (it contains its own 。), replaced by a placeholder, and turned into a
# "speaker said ..." event when followed by と + a speech verb.
_QUOTE = re.compile(r"「([^「」]*)」", re.S)
_DANGLING_QUOTE = re.compile(r"「([^「」\n]*)(?=\n|$)")
_QUOTE_PH = re.compile(r"\x01(\d+)\x01")
_SPEECH_STEMS = {
    "言っ": "言う", "云っ": "言う", "いっ": "言う", "言い": "言う", "云い": "言う",
    "いい": "言う", "もうし": "申す", "申し": "申す", "もうす": "申す", "申す": "申す",
    "答え": "答える", "こたえ": "答える", "尋ね": "尋ねる", "たずね": "尋ねる",
    "きき": "聞く", "聞き": "聞く", "問い": "問う", "とい": "問う",
    "頼み": "頼む", "たのみ": "頼む", "たのん": "頼む", "叫び": "叫ぶ", "さけび": "叫ぶ",
    "つぶやい": "つぶやく", "わめい": "わめく", "呼び": "呼ぶ", "よび": "呼ぶ",
    "どなっ": "どなる", "どなり": "どなる", "ささやい": "ささやく", "きい": "聞く",
}
_SPEECH_AFTER = re.compile(
    r"^[、。\s]*と[、\s]*"
    r"(?:(?:いって|云って|言って)[、\s]*)?"           # 「…」と いって、V too
    rf"(?:([{KANJI}{KATAKANA}]|{JP}{{2,6}})(?:は|が)[、\s]*)?"   # 「…」と 熊が V
    r"(" + "|".join(sorted(_SPEECH_STEMS, key=len, reverse=True)) + r")")
_NP_MARKED = re.compile(f"^([{KANJI}{KATAKANA}]|{JP}{{2,8}}?)(は|が)")

# Case particles, longest first.  が/を are the reliable anchors; に/へ/から
# introduce obliques; は/も are the topic and are only matched at clause start
# (mid-run they are usually word-internal: もも, おも..., ...も).
STRONG_PARTICLES = ["から", "が", "を", "に", "へ", "で", "と"]
TOPIC_PARTICLES = ["は", "も"]
_PARTICLE_TAILS = ("が", "を", "に", "へ", "で", "と", "は", "も", "から", "の")
SUBJECT_MARKERS = ("が", "は", "も")
OBJECT_MARKERS = ("を",)

SENTENCE_END = "。！？、」』）\n 　"
CONNECTIVES = {"そして", "それから", "すると", "しかし", "けれども", "でも",
               "ところが", "やがて", "また", "その", "この", "あの", "ある",
               "むかし", "むかしむかし", "とても", "やがて", "ある日", "いつも"}
# body parts / faculties: in a 「Xは Yが <state>」 sensation clause (おなかがすく,
# のどがかわく, あたまがいたい) the が-noun Y is part of the predicate, not the
# agent -- the experiencer is the dropped topic.  Taking Y as the subject and
# threading it forward is how "おなか" ends up "saying" things three clauses later.
PREDICATE_GA_NOUNS = frozenset({
    "おなか", "はら", "のど", "むね", "せなか", "こし", "あたま",
    "きもち", "きぶん", "からだ"})
# common nouns that embed a particle character -- protect them from the scanner
PROTECTED_NOUN_HEADS = ("もも", "おに", "かに", "とり", "にわ", "には虫", "きのこ",
                        "はな", "はた", "はし", "はこ", "もり", "こども", "ともだち",
                        "でんち", "にもつ", "におい")
# leading adjectival / determiner / adverbial prefixes to strip off a noun
NOUN_PREFIX = re.compile(r"^(大きな|小さな|きれいな|りっぱな|元気な|かわいい|やさしい|"
                         r"わるい|いい|ある|その|この|あの|ひとつの|一つの|"
                         r"ひとりの|ふたりの|いっぴきの|いちわの|いっぽんの|としとった|"
                         r"たくさんの|おおくの|すべての|いくつかの|なんびきかの|"
                         r"まだ|もう|ずっと|やがて|すぐ|とても|いつも|きっと|"
                         r"でも|そして|それから|すると|しかし|ところが|また)")
# 「... という ...」: the noun before という names the entity that follows
TO_IU = re.compile(r"という")
# a clause that is only a topic NP ("ねこは、") carries no verb but does set the
# subject for the clauses that follow it.  は only: 「いっぴきも」「だれも」 are
# quantifier+も, not an entity.
BARE_TOPIC = re.compile(f"^({JP}{{2,}})は$")
NON_TOPIC_NOUNS = {"それ", "これ", "あれ", "どれ", "だれ", "なに", "みんな", "みな",
                   "ここ", "そこ", "あそこ", "いま", "あと", "つぎ"}
# a bare counter / quantifier ("いっぴき", "ひとつ", "ふたり") -- with も it means
# "(not) even one", never names an entity
COUNTER_WORD = re.compile(
    r"^(ひとつ|ふたつ|みっつ|よっつ|いくつ|ひとり|ふたり|さんにん|"
    r"(いっ|に|さん|よん|ろっ|なな|はっ)?[ぴひび]き|"
    r"いちわ|にわ|さんわ|いっぽん|にほん|いっこ|にこ)$")

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


def _protect_quotes(text: str) -> "tuple[str, list[str]]":
    """Replace every 「…」 span with a \x01N\x01 placeholder and collect the quote
    strings, so a quote's own 。 does not split the sentence and its words are not
    scanned as case-marked NPs."""
    quotes: list[str] = []

    def take(match: "re.Match") -> str:
        quotes.append(match.group(1).strip())
        return f"\x01{len(quotes) - 1}\x01"

    prev = None
    while prev != text:                          # inner-most first, for nesting
        prev = text
        text = _QUOTE.sub(take, text)
    return _DANGLING_QUOTE.sub(take, text), quotes


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
    subject_explicit: bool = field(default=True, compare=False)

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.verb}|{self.obj}"


_KANJI_KATA = re.compile(f"[{KATAKANA}{KANJI}]")


def _clean_noun(raw: str) -> str:
    noun = re.sub(f"[^{HIRAGANA}{KATAKANA}{KANJI}]", "", raw)
    for _ in range(3):                       # 「いっぴきのとしとったねずみ」 stacks two
        stripped = NOUN_PREFIX.sub("", noun)
        if stripped == noun or not stripped:
            break
        noun = stripped
    return noun


def _noun_ok(raw: str) -> bool:
    """A plausible NP head: >=2 chars, OR a single kanji/katakana (山, 川, 犬)."""
    noun = _clean_noun(raw)
    return len(noun) >= 2 or (len(noun) == 1 and bool(_KANJI_KATA.match(noun)))


_ADJ_MODIFIER = re.compile(
    r"^(ちいさい|ちいさな|おおきい|おおきな|わかい|としとった|としよりの|うつくしい|"
    r"きれいな|かわいい|やさしい|わるい|かなしい|うれしい|ずるい|おろかな|"
    r"あわれな|びんぼうな|かねもちの|ゆうめいな|しあわせな|ふしあわせな)")


# adnominal verb tails that unambiguously end a relative clause modifying the
# following noun ("あそびまわっていた+こうもり", "そこにいた+いたち")
_REL_CLAUSE_TAIL = ("ている", "ていた", "ていない", "てある", "でいる", "でいた",
                    "った", "いた", "えた", "きた", "した", "ない", "れる", "られる")


def _strip_modifier(noun: str) -> str:
    """「あそびまわっていたこうもり」-> 「こうもり」, 「ちいさい女の子」-> 「女の子」:
    drop a leading relative-clause verb or adjective so the head noun is the
    subject, not a sentence fragment.  Conservative -- only unambiguous tails, so
    a name like ももたろう is never split."""
    noun = _ADJ_MODIFIER.sub("", noun)
    for cut in range(len(noun) - 2, 1, -1):
        prefix, head = noun[:cut], noun[cut:]
        if not prefix.endswith(_REL_CLAUSE_TAIL):
            continue
        if head in ("こと", "もの", "ひと", "とき", "ところ", "ため"):
            continue
        # the head must look like a content noun, not the tail of one word
        if not (len(head) >= 3 or _KANJI_KATA.match(head)):
            continue
        _, conf = _dictionary_verb(prefix)
        if conf >= 0.6:
            return _clean_noun(head)
    return noun


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
                # が followed by an inflection kana is verb-internal (上がる, 転がる);
                # が at the very end of a fragment is a real subject marker
                if particle == "が" and nxt and nxt in "るりっられろ":
                    continue
                # で in でした/です/でしょう or after ん is copula/verb-internal
                if particle == "で" and ((nxt and nxt in "しす") or prev == "ん"):
                    continue
                # a known word spanning the particle -> not a boundary, but only
                # when `nxt` continues that word.  An induced "こうもりが" chunk
                # (ends in the particle, or nxt is empty / a verb start) must not
                # suppress the real subject marker.
                _spanning = [current[-2:] + particle + nxt, current[-1:] + particle + nxt]
                if (known_words and nxt and any(
                        w in known_words and not w.endswith(_PARTICLE_TAILS)
                        for w in _spanning)):
                    continue
                matched = particle
                break
            if matched is None and not seen_topic and not out:
                cleaned = _clean_noun(current)
                run = len(cleaned)
                # topic は only at the first NP after a >=2-char run; topic も
                # needs >=3 (も is far more often word-internal: もも, くも, ...).
                # 「いっぴきも」「だれも」: quantifier/deixis + も is "even", not a topic
                if cleaned not in NON_TOPIC_NOUNS and not COUNTER_WORD.match(cleaned):
                    for particle, need in (("は", 2), ("も", 3)):
                        if clause.startswith(particle, i) and run >= need:
                            matched, seen_topic = particle, True
                            break
            if matched:
                noun = _strip_modifier(_clean_noun(current))
                if noun and noun not in CONNECTIVES:
                    out.append((noun, matched))
                current = ""
                i += len(matched)
                verb_start = i
            else:
                current += clause[i]
                i += 1
    return out, verb_start


def _verb_after(core: str, verb_start: int) -> "tuple[str, float]":
    verb_surface = re.sub(r"^[、。「」『』（）\s]+", "", core[verb_start:])
    verb_run = JP_RUN.search(verb_surface)
    if not verb_run:
        return "", 0.0
    return _dictionary_verb(verb_run.group(0))


def _assemble_event(pairs: "list[tuple[str, str]]", verb: str, verb_conf: float,
                    recent_subject: str | None, sentence: str) -> JapaneseEvent | None:
    subject = obj = ""
    roles: dict[str, str] = {}
    suppressed_ga = False
    for noun, particle in pairs:
        if particle == "が" and not subject and noun in PREDICATE_GA_NOUNS:
            # 「おなかがすいた」: the が-noun is the predicate's theme, not the
            # agent -- keep it as a role and let the dropped topic be the subject
            roles.setdefault("が", noun)
            suppressed_ga = True
        elif particle in SUBJECT_MARKERS and not subject:
            subject = noun
        elif particle in OBJECT_MARKERS and not obj:
            obj = noun
        else:
            roles.setdefault(particle, noun)
    subject_explicit = bool(subject)
    if not subject:
        subject = recent_subject or ""
    if not subject and not suppressed_ga:
        return None
    confidence = round(min(verb_conf, 0.9 if obj or roles else 0.7), 3)
    if not subject_explicit:
        confidence = round(min(confidence, 0.6), 3)     # inherited / unknown subject
    return JapaneseEvent(subject=subject, verb=verb, obj=obj, confidence=confidence,
                         sentence=sentence.strip(), roles=roles,
                         subject_explicit=subject_explicit)


def extract_clause(sentence: str, recent_subject: str | None = None,
                   known_words: "set[str] | None" = None) -> JapaneseEvent | None:
    """One event from one clause -- the standalone / single-clause entry point.
    `extract_story` uses the chaining path below instead."""
    text = normalise_text(sentence).strip()
    if len(text) < 4:
        return None
    core = text.rstrip(SENTENCE_END)
    core = TO_IU.sub("", core, count=1) if TO_IU.search(core) else core
    pairs, verb_start = _split_particles(core, known_words)
    verb, verb_conf = _verb_after(core, verb_start)
    if not verb:
        return None
    return _assemble_event(pairs, verb, verb_conf, recent_subject, sentence)


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


def _bare_topic(clause: str) -> str | None:
    bare = BARE_TOPIC.match(clause)
    if not bare:
        return None
    topic = _strip_modifier(_clean_noun(bare.group(1)))
    if _noun_ok(bare.group(1)) and topic not in CONNECTIVES and topic not in NON_TOPIC_NOUNS:
        return topic
    return ""                                    # a topic-shaped clause, but not an entity


def _last_np_before(text: str) -> str:
    """The nearest preceding 「Xは」/「Xが」 noun -- the speaker of a quote.
    Prefer the は-topic of the closest clause; fall back to its が-subject."""
    for frag in reversed(re.split(r"[、\x01]", text)):
        marked = _NP_MARKED.findall(frag)                    # [(noun, marker), ...]
        for want in ("は", "が"):
            for noun, marker in marked:
                if marker != want:
                    continue
                noun = _strip_modifier(_clean_noun(noun))
                if _noun_ok(noun) and noun not in CONNECTIVES and noun not in NON_TOPIC_NOUNS:
                    return noun
    return ""


_SPEECH_TAIL = re.compile(r"^(まし(た|て)|ます|た|て|ながら|つつ|、)+")


def _speech_events(sentence: str, quotes: "list[str]", recent_subject: str | None
                   ) -> "tuple[list[JapaneseEvent], str]":
    """「…」と言いました -> speaker | 言う | roles={と: quote}.  Returns the events
    and the sentence with each consumed 「…」と<speech verb> span cut out."""
    events: list[JapaneseEvent] = []
    residual, cursor = [], 0
    for m in _QUOTE_PH.finditer(sentence):
        after = _SPEECH_AFTER.match(sentence[m.end():])
        if not after:
            continue
        idx = int(m.group(1))
        quote = quotes[idx] if 0 <= idx < len(quotes) else ""
        after_speaker = _strip_modifier(_clean_noun(after.group(1))) if after.group(1) else ""
        explicit_speaker = after_speaker or _last_np_before(sentence[:m.start()])
        speaker = explicit_speaker or (recent_subject or "")
        if not speaker or not _noun_ok(speaker):
            continue
        events.append(JapaneseEvent(
            subject=speaker, verb=_SPEECH_STEMS[after.group(2)], obj="",
            confidence=0.75, sentence=_QUOTE_PH.sub("", sentence).strip(),
            roles={"と": quote[:40]}, subject_explicit=bool(explicit_speaker)))
        # consume 「…」と<verb>(ました|て|ながら…) so the residue is not re-parsed
        end = m.end() + after.end()
        tail = _SPEECH_TAIL.match(sentence[end:])
        end += tail.end() if tail else 0
        residual.append(sentence[cursor:m.start()])
        cursor = end
    residual.append(sentence[cursor:])
    return events, _QUOTE_PH.sub("", "".join(residual))


def _sentence_events(sentence: str, recent_subject: str | None,
                     vocab: "set[str] | None", quotes: "list[str] | None" = None
                     ) -> list[JapaneseEvent]:
    """A multi-clause sentence is one predication chain: case-marked NPs from
    every 、-fragment feed the sentence's verbs.  A fragment with no verb of its
    own carries its NPs forward to the next verb (「…こうもりが、…おちて、…
    つかまってしまいました」-> こうもり is the subject of both おちる and つかまる)."""
    events: list[JapaneseEvent] = []
    if quotes:
        speech, sentence = _speech_events(sentence, quotes, recent_subject)
        events.extend(speech)
        if speech:
            recent_subject = speech[-1].subject
    sentence = _QUOTE_PH.sub("", sentence)        # drop any un-consumed placeholders
    core = TO_IU.sub("", sentence, count=1) if TO_IU.search(sentence) else sentence
    core = core.rstrip(SENTENCE_END)
    frags = [f.strip() for f in re.split(r"、", core) if f.strip()]
    carried: list[tuple[str, str]] = []          # NPs from verbless fragments
    for frag in frags:
        topic = _bare_topic(frag)
        if topic is not None:
            if topic:
                recent_subject = topic
            continue
        pairs, verb_start = _split_particles(frag, vocab)
        verb, verb_conf = _verb_after(frag, verb_start)
        if not verb:                             # pure NP fragment -> carry it on
            carried.extend(pairs)
            continue
        event = _assemble_event(carried + pairs, verb, verb_conf, recent_subject, sentence)
        if event and event.confidence >= 0.4:
            events.append(event)
            carried = []                         # consumed by a trusted predicate
            if event.subject:
                recent_subject = event.subject
        else:
            # this sub-clause's predicate is too weak to trust -- drop its
            # obliques, but a 「牛が…よりあつまって」 subject still belongs to the
            # sentence's main verb, so carry the subject markers forward
            carried = [(n, p) for n, p in carried + pairs if p in SUBJECT_MARKERS]
    return events


def extract_story(text: str, known_words: "set[str] | None" = None) -> list[JapaneseEvent]:
    """Ordered events for a whole story, threading the omitted subject across
    clauses and sentences.  Direct speech (「…」と言った) is pulled out first so a
    quote's own 。 does not split the sentence."""
    dequoted, quotes = _protect_quotes(normalise_text(text))
    vocab = known_words if known_words is not None else learn_word_vocabulary(
        _QUOTE_PH.sub("", dequoted))
    events: list[JapaneseEvent] = []
    recent_subject: str | None = None
    for raw in SENT_SPLIT.split(dequoted):
        sentence = raw.strip()
        if len(_QUOTE_PH.sub("", sentence)) < 4:
            continue
        for event in _sentence_events(sentence, recent_subject, vocab, quotes):
            events.append(event)
            if event.subject:
                recent_subject = event.subject
    return events
