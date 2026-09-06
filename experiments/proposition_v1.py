#!/usr/bin/env python3
"""Extract stative and relational propositions the action-event parser discards.

`narrative_event_v29` keeps only explicit action clauses; ~7,000 admitted
sentences are quarantined as `no_explicit_action` / `auxiliary_without_action`.
Most of those are copular or possessive clauses that carry the *conceptual*
content -- what an entity is like, what it has, where it is.  This module turns
them into `entity|relation|value` propositions:

  "The fox was hungry."          -> fox|is|hungry           (property)
  "The grapes were not ripe."    -> grapes|is-not|ripe      (negated property)
  "The fox had a cunning plan."  -> fox|has|plan            (possession)
  "The nest was in the tree."    -> nest|in|tree            (spatial relation)

Rule-based and inspectable, no pretrained model.  A proposition is only emitted
when the subject and the value are both contentful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from narrative_event_v29 import VERBS

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Irregular past participles that look like adjectives after a copula but are
# really passives ("the fox was seen", not a property).  Regular -ed/-ing/-en
# are caught morphologically.
PARTICIPLES = {
    "said", "made", "seen", "found", "gone", "done", "come", "become", "left",
    "told", "heard", "kept", "held", "felt", "met", "sold", "built", "sent",
    "spent", "lost", "won", "run", "begun", "brought", "bought", "caught",
    "taught", "thought", "sought", "fought", "put", "set", "cut", "let", "hit",
    "read", "led", "fed", "bred", "shed", "given", "taken", "known", "grown",
    "shown", "thrown", "drawn", "worn", "torn", "born", "sworn", "chosen",
    "frozen", "broken", "spoken", "stolen", "written", "driven", "risen",
    "fallen", "eaten", "beaten", "hidden", "bidden", "forbidden",
}
NON_SUBJECT = {"there", "here", "it", "he", "she", "they", "we", "you", "i",
               "this", "that", "these", "those", "what", "which", "who",
               "one", "some", "any", "none", "all", "each"}
# participial forms that really are stative adjectives ("the fox was tired")
STATE_ADJECTIVES = {
    "tired", "frightened", "pleased", "delighted", "excited", "worried",
    "surprised", "amazed", "astonished", "annoyed", "ashamed", "afraid",
    "alarmed", "charmed", "contented", "satisfied", "determined", "willing",
    "unwilling", "cunning", "charming", "amusing", "interesting", "pleasing",
    "loving", "daring", "tempting", "puzzled", "troubled", "wounded",
    "starved", "exhausted", "delighted", "grieved", "vexed", "enraged",
}

COPULA = {"is", "are", "was", "were", "be", "been", "being", "am", "'s", "'re"}
HAVE = {"have", "has", "had"}
NEGATION = {"not", "never", "no", "n't", "neither", "nor"}
LOCATIVE = {"in", "on", "at", "near", "under", "over", "beside", "behind",
            "within", "among", "amongst", "inside", "outside", "above", "below",
            "beneath", "atop", "by", "against", "around", "amid"}
ARTICLES = {"a", "an", "the", "this", "that", "these", "those"}
POSSESSIVES = {"his", "her", "its", "their", "my", "our", "your"}
DEGREE = {"very", "quite", "so", "too", "rather", "most", "more", "less", "as",
          "much", "still", "even", "always", "already", "now", "then", "there"}
# closed-class / non-value words that must never be a property or relation value
NON_VALUE = ARTICLES | POSSESSIVES | DEGREE | NEGATION | COPULA | HAVE | {
    "and", "or", "but", "if", "of", "to", "for", "with", "from", "who", "which",
    "what", "when", "where", "why", "how", "here", "one", "some", "any", "all",
    "it", "he", "she", "they", "him", "them", "we", "you", "i",
}
SUBJECT_STOP = {"and", "or", "but", "of", "to", "in", "on", "at", "by", "for",
                "with", "from", "who", "which", "that", "as", "than", "when",
                "where", "while", "because", "though", "although", "if"}


@dataclass(frozen=True)
class Proposition:
    subject: str
    relation: str          # "is" | "is-not" | "has" | a locative preposition
    value: str
    polarity: str          # "positive" | "negative"
    sentence: str

    @property
    def key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.value}"


def _is_verby(word: str) -> bool:
    if word in STATE_ADJECTIVES:
        return False
    return (word in VERBS or word in PARTICIPLES
            or (word.endswith(("ed", "ing", "en")) and len(word) > 4))


def _subject(tokens: list[str], verb_index: int, subject_hint: str | None) -> str | None:
    pre = tokens[:verb_index]
    cut = next((i for i, w in enumerate(pre) if w in SUBJECT_STOP), None)
    if cut is not None:
        pre = pre[:cut]
    candidates = [w for w in pre if w not in ARTICLES | POSSESSIVES and not w.endswith("ly")]
    if not candidates:
        return None
    head = candidates[-1].split("'", 1)[0]
    if head in {"he", "she", "it", "they", "we"} or (candidates[-1] in NON_SUBJECT):
        return subject_hint
    if len(head) < 3 or head in NON_SUBJECT or _is_verby(head):
        return None
    return head


def _value_after(tokens: list[str], start: int, allow_verby: bool = False) -> tuple[str | None, str]:
    """First contentful (non-verb) word after `start`, plus polarity."""
    polarity = "positive"
    index = start
    while index < len(tokens) and tokens[index] in NEGATION | DEGREE:
        if tokens[index] in NEGATION:
            polarity = "negative"
        index += 1
    while index < len(tokens):
        word = tokens[index].split("'", 1)[0]
        if (len(word) >= 3 and word not in NON_VALUE and not word.endswith("ly")
                and (allow_verby or not _is_verby(word))):
            return word, polarity
        index += 1
    return None, polarity


def extract_proposition(sentence: str, subject_hint: str | None = None) -> Proposition | None:
    tokens = [w.lower() for w in WORD.findall(sentence)]
    if len(tokens) < 3 or len(tokens) > 20:
        return None

    # The copula/verb must sit near the start: a simple stative clause has its
    # subject early, and a late copula usually means we are inside a relative
    # or subordinate clause and would grab the wrong noun.
    copula_index = next((i for i, w in enumerate(tokens[1:7], 1) if w in COPULA), None)
    have_index = next((i for i, w in enumerate(tokens[1:7], 1) if w in HAVE), None)
    # "was <participle>" / "was <participle> by ..." is a passive or a
    # progressive, not a property -- unless the participle is a stative adjective
    if copula_index is not None:
        after = tokens[copula_index + 1:copula_index + 3]
        skip = [w for w in after if w not in NEGATION | DEGREE][:1]
        if skip and _is_verby(skip[0]):
            return None

    if copula_index is not None:
        subject = _subject(tokens, copula_index, subject_hint)
        if not subject:
            return None
        # spatial relation: copula + (negation) + locative preposition + noun
        rest = tokens[copula_index + 1:]
        prep_offset = next((k for k, w in enumerate(rest[:3]) if w in LOCATIVE), None)
        if prep_offset is not None:
            noun, polarity = _value_after(rest, prep_offset + 1)
            if noun:
                return Proposition(subject, rest[prep_offset], noun, polarity, sentence)
        value, polarity = _value_after(tokens, copula_index + 1)
        # a copular complement that is a verb form is a passive, not a property
        if value and value != subject and not _is_verby(value):
            relation = "is-not" if polarity == "negative" else "is"
            return Proposition(subject, relation, value, polarity, sentence)
        return None

    if have_index is not None:
        subject = _subject(tokens, have_index, subject_hint)
        if not subject:
            return None
        rest = tokens[have_index + 1:]
        skip = [w for w in rest[:2] if w not in NEGATION | DEGREE][:1]
        if skip and _is_verby(skip[0]):
            return None                       # "had gone", "has done" -> perfect tense
        polarity = "negative" if any(w in NEGATION for w in rest[:3]) else "positive"
        # possession takes the NP head: the last contentful noun in the object span
        span = [w.split("'", 1)[0] for w in rest[:5]
                if len(w) >= 3 and w not in NON_VALUE and not w.endswith("ly")
                and not _is_verby(w)]
        value = span[-1] if span else None
        if value and value != subject:
            relation = "has-not" if polarity == "negative" else "has"
            return Proposition(subject, relation, value, polarity, sentence)
    return None


def extract_document_propositions(sentences: list[str],
                                  subject_hints: "list[str | None] | None" = None
                                  ) -> list[Proposition]:
    out = []
    for index, sentence in enumerate(sentences):
        hint = subject_hints[index] if subject_hints and index < len(subject_hints) else None
        proposition = extract_proposition(sentence, hint)
        if proposition:
            out.append(proposition)
    return out
