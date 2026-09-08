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
#   3 -> direct-speech (「…」と言った) events; te-form subject carries forward
#   4 -> verb normalisation: negative-past, bare te-form, clause-tail stripping,
#        った defaults to る; single-kanji topic は
#   5 -> compound verbs (〜ておる, 〜ながら), copula ではなかった, ことができる,
#        んだ -> ぶ/む/ぬ, 考える/思う in the table; adverbs out of the subject slot
PARSER_VERSION = 6      # v6: conjunctions / sentence adverbs rejected as subjects

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
               "むかし", "むかしむかし", "とても", "やがて", "ある日", "いつも",
               "いろいろ", "だんだん", "そっと", "しずかに", "きゅうに", "にわかに",
               "しばらく", "まもなく", "すっかり", "ちっとも", "とうとう", "しまいに",
               "いきなり", "だいぶ", "もっと", "ずいぶん", "たいそう", "たいへん"}
# Conjunctions, sentence adverbs, connective fragments and (historical-kana)
# variants that the case scanner otherwise admits as a bare topic / subject --
# e.g. 「けれども、…」 splits into topic "けれど" + particle も.  A token here can
# never be the SUBJECT of an event; it is dropped (the dropped topic threads
# through instead).  Only unambiguous function words -- nothing that is also a
# common noun (heat, water, morning...).
NON_SUBJECT = {
    "けれど", "けど", "だけど", "だが", "しかも", "それに", "つまり", "だから",
    "それで", "そこで", "ですから", "および", "または", "あるいは", "ないし",
    "もし", "もしも", "たとえ", "まるで", "ちょうど", "やはり", "やっぱり",
    "きっと", "たぶん", "おそらく", "まさか", "けっして", "ぜひ", "どうか",
    "なぜ", "なぜなら", "どうして", "いったい", "はたして", "せっかく",
    "こう", "そう", "ああ", "どう",
    "たうたう", "とうとう", "しまひに", "だんだんに", "そのうち", "やうやう",
}
# body parts / faculties: in a 「Xは Yが <state>」 sensation clause (おなかがすく,
# のどがかわく, あたまがいたい) the が-noun Y is part of the predicate, not the
# agent -- the experiencer is the dropped topic.  Taking Y as the subject and
# threading it forward is how "おなか" ends up "saying" things three clauses later.
PREDICATE_GA_NOUNS = frozenset({
    "おなか", "はら", "のど", "むね", "せなか", "こし", "あたま",
    "きもち", "きぶん", "からだ",
    # formal / idiomatic nouns that head a set phrase (しかたがない, わけがない,
    # しようがない) -- the が-noun is not an agent
    "しかた", "しよう", "わけ", "はず", "しょう"})
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
    "できました": "できる", "できません": "できる", "できませんでした": "できる",
    "できない": "できる", "できなかった": "できる", "でき": "できる", "できて": "できる",
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
    # frequent verbs the scanner otherwise mis-splits (考える at が, 思う as 思る…)
    "考えた": "考える", "かんがえた": "考える", "考えました": "考える", "かんがえました": "考える",
    "考える": "考える", "かんがえる": "考える", "考えて": "考える", "かんがえて": "考える",
    "思った": "思う", "おもった": "思う", "思いました": "思う", "おもいました": "思う",
    "思う": "思う", "おもう": "思う", "思って": "思う", "おもって": "思う",
    "答えた": "答える", "こたえた": "答える", "答えました": "答える", "こたえました": "答える",
    "わかった": "わかる", "わかりました": "わかる", "分かった": "わかる",
    "わかる": "わかる", "わからない": "わかる", "わからなかった": "わかる", "わかって": "わかる",
    "言って": "言う", "いって": "言う", "云った": "言う", "云いました": "言う", "云う": "言う",
    "聞いた": "聞く", "きいた": "聞く", "聞きました": "聞く", "ききました": "聞く",
}
# godan: i-row (ます-stem last kana) -> dictionary u-row
I_TO_U = {"い": "う", "き": "く", "ぎ": "ぐ", "し": "す", "ち": "つ",
          "に": "ぬ", "ひ": "ふ", "び": "ぶ", "み": "む", "り": "る"}
# past-tense godan endings: surface tail -> (drop, dictionary tail).  For った the
# ending is ambiguous (買う/待つ/取る all -> った); default to る (つかまる, とまる,
# かかる, わかる are far more common in these stories than the odd new う-verb) and
# list the frequent う-verbs whose stem ends the surface before った.
GODAN_PAST = {"った": ["る", "つ", "う"], "いた": ["く"], "いだ": ["ぐ"],
              "した": ["す"], "んだ": ["む", "ぶ", "ぬ"]}
_GODAN_U_STEMS = ("思", "おも", "笑", "わら", "使", "つか", "歌", "うた", "買", "か",
                  "会", "合", "あ", "手伝", "てつだ", "もら", "はら", "うしな", "した",
                  "すく", "とりあ", "であ", "い")
# …んだ that is really a ぶ-verb (遊ぶ/飛ぶ/呼ぶ/喜ぶ/運ぶ), or 死ぬ
_GODAN_BU_STEMS = ("あそ", "遊", "と", "飛", "よ", "呼", "はこ", "運", "よろこ", "喜",
                   "ころ", "転", "むす", "結", "えら", "選")
_GODAN_NU_STEMS = ("し", "死")


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
                    r"しま(った|いました|う)|"
                    r"お(いた|きました|り|りました|ります|る|く)|"      # 〜ておる/ておく
                    r"あ(る|った|ります|りました)|"                    # 〜てある
                    r"み(た|ました|る))$")
_NAGARA = re.compile(rf"^{JP}{{1,}}?(ながら|つつ)(?=.)")     # 見ながら言う -> drop 見ながら
# clause-final nominalisers / conjunctions that hang off a finished verb
_VERB_TAIL = re.compile(
    r"(の(だ|です|である|でした)?|(?<=[うくぐすつぬぶむる])ん(だ|です)|"
    r"(?<=[うくぐすつぬぶむる])んだと|"
    r"から|ので|のに|けれど[も]?|"
    r"(もの|の|わけ)?で(は|も)?(ない|なかった|ありません|ありませんでした)|"   # …ものではなかった
    r"のである|のでした|ということ)$")
_NEG_PAST = re.compile(r"な(かった|かっ)(ら|ので|のです|のである|から|けれど[も]?|り)?$")


def _dictionary_verb(surface: str) -> tuple[str, float]:
    """(dictionary form, confidence)."""
    surface = surface.strip("。、！？「」『』（）　 \n")
    if not surface:
        return "", 0.0
    # 泳ぐことができた -> 泳ぐ (keep the content verb, drop the potential auxiliary)
    surface = re.sub(r"こと(が|は|も)?でき.*$", "", surface) or surface
    surface = re.sub(r"^(ことが|のが|ことは|のは|ことも|わけには)", "", surface)
    surface = re.sub(r"^[はがを](?=でき|いられ|おられ)", "", surface)   # …ことはできない
    surface = _NAGARA.sub("", surface, count=1)   # 見ながら言いました -> 言いました
    for _ in range(2):                           # 受けたのである -> 受けた
        stripped = _VERB_TAIL.sub("", surface)
        if stripped == surface or len(stripped) < 2:
            break
        surface = stripped
    surface = _NEG_PAST.sub("ない", surface)      # できなかった -> できない
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
            if not surface.endswith(tail):
                continue
            root = surface[:-len(tail)]
            if tail == "った" and root.endswith(_GODAN_U_STEMS):
                return root + "う", 0.7
            if tail == "んだ" and root.endswith(_GODAN_BU_STEMS):
                return root + "ぶ", 0.7
            if tail == "んだ" and root.endswith(_GODAN_NU_STEMS):
                return root + "ぬ", 0.75
            return root + options[0], 0.6
        if surface.endswith("た"):               # ichidan past: 食べた -> 食べる
            return surface[:-1] + "る", 0.65
    # bare te-form that TE_AUX did not catch: おちて -> おちる, 見て -> 見る
    if surface.endswith("て") and not surface.endswith(("って", "いて", "して")):
        return surface[:-1] + "る", 0.6
    if surface.endswith("って"):                  # 走って -> 走る (default る, not う)
        return surface[:-2] + "る", 0.5
    if surface.endswith(("いで", "んで")):
        tail = {"いで": "ぐ", "んで": "む"}[surface[-2:]]
        return surface[:-2] + tail, 0.5
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
                # が at the very end of a fragment is a real subject marker.  が+え
                # is verb-internal only when what follows inflects (かんがえた,
                # きこえる) -- not before a noun (きつねがえさを).
                if particle == "が" and nxt and (
                        nxt in "るりっられろ"
                        or (nxt == "え" and clause[i + 2:i + 3] in ("る", "た", "て", "ま", "よ", "な"))):
                    continue
                # で in でした/です/でしょう, で+は (copula では), or after ん is
                # copula / verb-internal, not a locative
                if particle == "で" and ((nxt and nxt in "しすは") or prev == "ん"):
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
                # topic は after a >=2-char run, OR a single kanji/katakana noun
                # (犬は, 熊は); topic も needs >=3 (も is far more often
                # word-internal).  「いっぴきも」「だれも」: quantifier/deixis + も
                # is "even", not a topic.
                single_kanji = run == 1 and bool(_KANJI_KATA.match(cleaned))
                if (cleaned not in NON_TOPIC_NOUNS and cleaned not in NON_SUBJECT
                        and not COUNTER_WORD.match(cleaned)):
                    for particle, need in (("は", 1 if single_kanji else 2), ("も", 3)):
                        if clause.startswith(particle, i) and run >= need:
                            matched, seen_topic = particle, True
                            break
            if matched:
                noun = _strip_modifier(_clean_noun(current))
                if noun and noun not in CONNECTIVES and noun not in NON_SUBJECT:
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
        elif particle in SUBJECT_MARKERS and not subject and noun not in NON_SUBJECT:
            subject = noun
        elif particle in OBJECT_MARKERS and not obj and noun not in NON_SUBJECT:
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
    if (_noun_ok(bare.group(1)) and topic not in CONNECTIVES
            and topic not in NON_TOPIC_NOUNS and topic not in NON_SUBJECT):
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
                if (_noun_ok(noun) and noun not in CONNECTIVES
                        and noun not in NON_TOPIC_NOUNS and noun not in NON_SUBJECT):
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


def extract_story(text: str, known_words: "set[str] | None" = None,
                  use_teacher: bool = False) -> list[JapaneseEvent]:
    """Ordered events for a whole story, threading the omitted subject across
    clauses and sentences.  Direct speech (「…」と言った) is pulled out first so a
    quote's own 。 does not split the sentence.

    `use_teacher=True` runs a disclosed morphological analyser (evidence score 0)
    over the same sentences afterwards and lets it correct verb dictionary forms
    and strip relative-clause fragments from subjects.  It is OFF by default:
    the frozen benchmarks, the character RNN, and boundary induction only ever
    see the heuristic parse (ARCHITECTURE.md invariants 1, 16, 17)."""
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
    if use_teacher:
        events = refine_with_teacher(events)
    return events


_TEACHER_AUX_VERBS = {"いる", "おる", "ある", "くる", "いく", "行く", "来る", "しまう",
                      "みる", "見る", "おく", "くれる", "もらう", "あげる", "ゆく",
                      "なる", "する", "だ", "です"}


def _teacher_main_verb(analysis, heuristic_verb: str) -> str | None:
    """The teacher's dictionary form for the SAME verb the heuristic found --
    matched by a shared prefix.  No confident match -> keep the heuristic verb
    (one event's `.sentence` is the whole, possibly multi-verb sentence, so a
    positional fallback would often pick the wrong clause)."""
    hv = heuristic_verb or ""
    if len(hv) < 2:
        return None
    for m in analysis.morphemes:
        if not m.is_verb:
            continue
        if (m.surface[:2] == hv[:2] or m.base[:2] == hv[:2]
                or hv.startswith(m.surface[:3]) or m.base in hv or hv in m.base):
            return m.base
    return None


def _teacher_noun_head(analysis, subject: str) -> str:
    """Keep only the trailing noun run of `subject` (drops あそびまわっていた in
    あそびまわっていたこうもり) using the analyser's part-of-speech tags."""
    if not subject or len(subject) < 3:
        return subject
    acc, head = "", []
    for m in analysis.morphemes:
        if not acc and not subject.startswith(m.surface):
            continue
        if acc and not subject.startswith(acc + m.surface):
            break
        acc += m.surface
        if m.pos in ("名詞", "代名詞"):
            head.append(m.surface)
        elif m.pos in ("動詞", "助動詞", "形容詞", "助詞"):
            head = []
        if acc == subject:
            break
    joined = "".join(head)
    return joined if len(joined) >= 2 else subject


def _refine_dicts(events: list[dict]) -> list[dict]:
    try:
        import morphology_teacher as _mt
    except Exception:
        return events
    teacher = _mt.get_teacher()
    if not teacher.available():
        return events
    analyses: dict[str, object] = {}
    out: list[dict] = []
    for e in events:
        sent = e.get("sentence") or ""
        if sent not in analyses:
            analyses[sent] = teacher.analyse(sent)
        a = analyses[sent]
        e = dict(e)
        if a is not None:
            verb = _teacher_main_verb(a, e.get("verb", "")) or e.get("verb", "")
            subject = _teacher_noun_head(a, e.get("subject", "")) if e.get("subject") else e.get("subject", "")
            if verb != e.get("verb") or subject != e.get("subject"):
                # a score-0 proposal re-reads the SAME textual evidence -- it may
                # supply a cleaner surface form but must not raise confidence
                # (ARCHITECTURE.md invariant 10); cap it at the heuristic value
                e["verb"], e["subject"] = verb, subject
                e["confidence"] = round(min(e.get("confidence") or 0.5, 0.6), 3)
                e["teacher_refined"] = True
        out.append(e)
    return out


def refine_with_teacher(events: list[JapaneseEvent]) -> list[JapaneseEvent]:
    """Correct verb base forms and subject fragments using a disclosed
    morphological analyser.  No-op when none is available -- the analyser's
    output is a proposal (evidence score 0), never authority."""
    refined = _refine_dicts([e.__dict__ for e in events])
    return [JapaneseEvent(subject=d.get("subject", ""), verb=d.get("verb", ""),
                          obj=d.get("obj", ""), confidence=d.get("confidence", 0.5),
                          sentence=d.get("sentence", ""), roles=d.get("roles") or {},
                          subject_explicit=d.get("subject_explicit", True))
            for d in refined]


def refine_event_dicts(events: list[dict]) -> list[dict]:
    """Teacher refinement for events already in dict form (the reading loop's
    cached events)."""
    return _refine_dicts(events)
