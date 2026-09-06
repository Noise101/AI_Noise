#!/usr/bin/env python3
"""Within-document coreference: give an event sequence one protagonist thread.

Rule-based and inspectable, no pretrained model.  For each source's ordered
sentence list it tracks the entities introduced so far and, per sentence,
reports which known entity the subject (and trailing object) position refers to:

  * proper noun / "a|an <noun>"       -> a new entity
  * "the <noun>" / bare repeated noun  -> the most recent entity with that head
  * pronoun (he/she/it/they/him/...)   -> the most recent number/animacy-compatible
                                          entity inside a short recency window

The extractor already maps a subject pronoun to its `recent_subject` argument;
this module supplies a *better* antecedent than "the last accepted subject":
one with number/animacy agreement, chosen from a recency window, that survives
an intervening sentence whose own extraction failed.  It does NOT do
coreference-by-description ("the poor creature" -> "fox") -- that needs lexical
knowledge the core learner does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

ARTICLES = {"a", "an", "the"}
DEFINITE = {"the", "this", "that", "these", "those"}
INDEFINITE = {"a", "an"}
SUBJECT_PRONOUNS = {"he", "she", "it", "they", "we"}
OBJECT_PRONOUNS = {"him", "her", "it", "them", "us"}
PRONOUNS = SUBJECT_PRONOUNS | OBJECT_PRONOUNS
NUMBER = {"he": "sg", "she": "sg", "it": "sg", "him": "sg", "her": "sg",
          "they": "pl", "them": "pl", "we": "pl", "us": "pl"}
ANIMATE_PRONOUNS = {"he", "she", "him", "her"}
NEUTER_PRONOUNS = {"it"}
# words that are never an entity head: closed-class + sentence-initial
# adverbs / discourse markers / temporals that the greedy first-noun scan
# would otherwise register as bogus entities ("Oftentimes he ...", "Anyhow ...")
NON_ENTITY = {
    "a", "an", "the", "this", "that", "these", "those", "and", "or", "but", "if",
    "so", "then", "there", "here", "not", "no", "very", "too", "as", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "into", "over", "under", "up",
    "out", "off", "down", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "whether", "while", "because", "although", "though", "unless",
    "is", "are", "was", "were", "be", "been", "being", "am", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "can", "may", "might",
    "must", "one", "some", "any", "all", "each", "every", "many", "much", "more",
    "most", "few", "several", "both", "either", "neither", "none",
    "his", "her", "their", "its", "my", "our", "your", "mine", "yours",
    "oftentimes", "anyhow", "anyway", "however", "moreover", "furthermore",
    "nevertheless", "therefore", "thus", "hence", "meanwhile", "afterwards",
    "afterward", "presently", "immediately", "suddenly", "finally", "eventually",
    "perhaps", "maybe", "certainly", "surely", "indeed", "besides", "instead",
    "sometimes", "always", "never", "often", "seldom", "rarely", "once", "twice",
    "soon", "now", "today", "yesterday", "tomorrow", "tonight", "again",
    "here", "there", "everywhere", "nowhere", "somewhere", "thereupon", "hereupon",
    "next", "first", "second", "third", "last", "later", "earlier", "before",
    "after", "since", "until", "till", "yet", "still", "also", "even", "just",
    "only", "quite", "rather", "almost", "nearly", "well", "long", "far",
    "oh", "ah", "alas", "yes", "nay", "lo", "behold", "hark", "hush",
    "deep", "deeply", "high", "highly", "low", "close", "near", "wide", "amid",
    "amongst", "atop", "beneath", "within", "throughout", "across", "along",
    "around", "behind", "beside", "between", "toward", "towards", "against",
    "beyond", "above", "below", "outside", "inside", "past", "through",
}
ANIMATE_HINTS = {
    "man", "woman", "boy", "girl", "child", "children", "king", "queen", "prince",
    "princess", "father", "mother", "son", "daughter", "brother", "sister", "friend",
    "wife", "husband", "people", "men", "women", "farmer", "hunter", "shepherd",
    "fox", "wolf", "lion", "hare", "rabbit", "mouse", "rat", "bird", "crow", "dog",
    "cat", "bear", "goat", "sheep", "lamb", "horse", "ass", "donkey", "frog", "snake",
    "fish", "ant", "bee", "eagle", "owl", "hen", "cock", "duck", "swan", "deer",
    "tortoise", "turtle", "monkey", "elephant", "tiger", "cow", "bull", "pig",
    "master", "servant", "traveller", "traveler", "stranger", "neighbour", "neighbor",
    "god", "spirit", "fairy", "witch", "giant", "dwarf", "sailor", "soldier", "doctor",
}


@dataclass
class Entity:
    canonical: str
    number: str          # "sg" | "pl"
    animate: bool
    last_mention: int    # sentence index


@dataclass
class DocumentCoref:
    subject_hints: list[str | None]           # per sentence: resolved subject entity
    object_hints: list[str | None]            # per sentence: resolved trailing-object entity
    resolutions: int = 0                       # links made (pronoun + definite)
    new_entities: int = 0
    sentences: int = 0
    entity_names: list[str] = field(default_factory=list)


def _head_noun(tokens: list[str], start: int) -> tuple[str | None, int]:
    """First plausible noun at/after `start`, skipping determiners and adjectives
    up to a small span.  Returns (noun, index) or (None, -1)."""
    for index in range(start, min(len(tokens), start + 4)):
        word = tokens[index].split("'", 1)[0]        # strip possessive "'s"
        if not word or word in NON_ENTITY or word.endswith("ly"):
            continue
        return word, index
    return None, -1


def _is_animate(noun: str) -> bool:
    base = noun[:-1] if noun.endswith("s") and len(noun) > 3 else noun
    return noun in ANIMATE_HINTS or base in ANIMATE_HINTS


def _number_of(noun: str) -> str:
    if noun.endswith(("s", "es")) and not noun.endswith(("ss", "us", "is")) and len(noun) > 3:
        return "pl"
    return "sg"


def resolve_document(sentences: list[str], recency_window: int = 3) -> DocumentCoref:
    entities: list[Entity] = []
    by_head: dict[str, Entity] = {}
    subject_hints: list[str | None] = []
    object_hints: list[str | None] = []
    resolutions = new_entities = 0

    # Proper nouns: words seen capitalised at a non-initial position anywhere in
    # the document (sentence-initial capitalisation is not informative).
    proper = set()
    for sentence in sentences:
        for token in WORD.findall(sentence)[1:]:
            if token[:1].isupper():
                proper.add(token.lower().split("'", 1)[0])

    def most_recent_compatible(pronoun: str, before: int) -> Entity | None:
        want_number = NUMBER.get(pronoun, "sg")
        candidates = [e for e in entities
                      if e.number == want_number and 0 < before - e.last_mention <= recency_window]
        if pronoun in ANIMATE_PRONOUNS:
            candidates = [e for e in candidates if e.animate]      # no fallback: he/she need animate
        elif pronoun in NEUTER_PRONOUNS:
            neuter = [e for e in candidates if not e.animate]
            candidates = neuter or candidates
        return max(candidates, key=lambda e: e.last_mention, default=None)

    def register(noun: str, index: int, animate: bool = False) -> Entity:
        nonlocal new_entities
        entity = Entity(canonical=noun, number=_number_of(noun),
                        animate=animate or _is_animate(noun), last_mention=index)
        entities.append(entity)
        by_head[noun] = entity
        new_entities += 1
        return entity

    for index, sentence in enumerate(sentences):
        raw_tokens = WORD.findall(sentence)
        tokens = [w.lower().split("'", 1)[0] for w in raw_tokens]
        subject_entity: str | None = None
        object_entity: str | None = None

        # --- subject position: first noun phrase / pronoun ---
        # Skip a leading run of adverbs / discourse markers ("Oftentimes he ...").
        start = 0
        while start < len(tokens) and tokens[start] in NON_ENTITY and tokens[start] not in ARTICLES | DEFINITE:
            start += 1
        if start < len(tokens):
            first = tokens[start]
            if first in SUBJECT_PRONOUNS:
                match = most_recent_compatible(first, index)
                if match:
                    match.last_mention = index
                    subject_entity = match.canonical
                    resolutions += 1
            else:
                determiner = first if first in ARTICLES | DEFINITE else None
                noun, noun_index = _head_noun(tokens, start + 1 if determiner else start)
                # A bare noun (no determiner) is trusted as an entity when it is
                # animate, a known proper noun, or capitalised here (a leading
                # adverb was already skipped, so a capitalised word at `start` is
                # usually a name -- an abstract noun occasionally leaks in, but
                # the animacy filter keeps it out of he/she resolution).
                bare_ok = (determiner is not None
                           or (noun is not None and noun_index < len(raw_tokens)
                               and (_is_animate(noun) or noun in proper
                                    or raw_tokens[noun_index][:1].isupper())))
                if noun and noun_index >= 0 and bare_ok:
                    proper_here = (noun in proper
                                   or (noun_index < len(raw_tokens)
                                       and raw_tokens[noun_index][:1].isupper()))
                    if determiner in INDEFINITE or noun not in by_head:
                        subject_entity = register(
                            noun, index, animate=proper_here and noun not in NON_ENTITY).canonical
                    else:
                        by_head[noun].last_mention = index
                        subject_entity = by_head[noun].canonical
                        if determiner in DEFINITE:
                            resolutions += 1

        # --- object noun phrases: track known / animate / proper referents ---
        for position in range(1, len(tokens)):
            if tokens[position - 1] not in ARTICLES | DEFINITE:
                continue
            head, head_index = _head_noun(tokens, position)
            if not head or head == subject_entity:
                continue
            if head in by_head:
                by_head[head].last_mention = index
            elif _is_animate(head) or head in proper or (
                    head_index < len(raw_tokens) and raw_tokens[head_index][:1].isupper()):
                register(head, index)
        # trailing referent: the last object pronoun, else the last definite NP
        for position in range(len(tokens) - 1, 0, -1):
            word = tokens[position]
            if word in OBJECT_PRONOUNS:
                match = most_recent_compatible(word, index)
                if match and match.canonical != subject_entity:
                    match.last_mention = index
                    object_entity = match.canonical
                    resolutions += 1
                break
            if word in by_head and tokens[position - 1] in DEFINITE:
                object_entity = by_head[word].canonical
                break

        subject_hints.append(subject_entity)
        object_hints.append(object_entity)

    return DocumentCoref(subject_hints=subject_hints, object_hints=object_hints,
                         resolutions=resolutions, new_entities=new_entities,
                         sentences=len(sentences),
                         entity_names=[e.canonical for e in entities])
