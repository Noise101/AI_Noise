#!/usr/bin/env python3
"""A knowledge -> capability loop over what the reading loop has learned.

The reading loop accumulates knowledge (word-meaning beliefs, parsed events)
but reading more does not, on its own, make Noise reason better.  This module
adds the missing loop:

    retrieve related memory
        -> abstract a reusable rule
        -> generate a problem it can auto-grade
        -> reason to an answer (recorded step by step)
        -> self-evaluate the reasoning
        -> store the attempt as an experience (success AND failure)
        -> next attempt consults the correction

Everything is grounded in Noise's own verified data: a problem is only made
when its answer can be derived from a belief the reading loop already tested,
the event order of a specific book, or the coarse ontology.  No LLM: the
reasoner is explicit retrieve/match/compose steps (ARCHITECTURE invariants
10, 13).  Capability is not "how many rules" -- it is measured on a frozen
held-out problem set by `capability_probe_v1` (invariant 16).

Toggle with AI_NOISE_COGNITION=0.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

VERSION = 1
PROBLEMS_PER_CYCLE = 8
MAX_REASONING_STEPS = 12
RULE_MIN_BOOKS = 2
RULE_MIN_SUPPORT = 3
RULE_REUSABLE_BOOKS = 3
RULE_REUSABLE_SUPPORT = 6
EXPERIENCE_CAP = 4000
RULES_CAP = 3000
HISTORY_CAP = 400

COARSE = ("生き物", "人", "植物", "食べ物", "道具", "場所", "自然物", "出来事", "気持ち")
# verbs that carry no information about the subject's kind -- every entity does them
_VACUOUS_VERBS = ("ある", "いる", "なる", "する", "くる", "来る", "いく", "行く", "みる",
                  "見る", "おもう", "思う", "いう", "言う", "できる", "しまう", "です",
                  "だる", "れる", "られる")


def _informative_verb(v: str) -> bool:
    v = (v or "").strip()
    return bool(v) and len(v) >= 2 and not any(v == s or v.startswith(s) for s in _VACUOUS_VERBS)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hid(*parts: str) -> str:
    return hashlib.sha256(" ".join(str(p) for p in parts).encode()).hexdigest()[:16]


def enabled() -> bool:
    return os.environ.get("AI_NOISE_COGNITION") != "0"


# --------------------------------------------------------------------------
# shared context -- the modules talk only through this, never call each other
# --------------------------------------------------------------------------
@dataclass
class CognitiveContext:
    cycle: int
    concept: str = ""
    concept_genus: str = ""
    events: list = field(default_factory=list)        # events of the book just read
    book_id: str = ""
    wm: dict = field(default_factory=dict)            # word-meaning state (read-only)
    heur_store: dict = field(default_factory=dict)    # bid -> [events]
    shelf: dict = field(default_factory=dict)
    retrieved: list = field(default_factory=list)     # [{memory_id, kind, content, relevance, source, created, used_count}]
    abstractions: list = field(default_factory=list)  # rule ids touched this pass
    problems: list = field(default_factory=list)
    solved: list = field(default_factory=list)        # [{pid, level, type, answer, gold, correct, confidence, steps, evaluation}]
    experiences: list = field(default_factory=list)
    log: list = field(default_factory=list)

    def trace(self, msg: str) -> None:
        self.log.append(msg)


class CognitiveModule:
    """Common shape.  `process` reads and writes the context in place."""
    name = "module"

    def can_handle(self, ctx: CognitiveContext) -> bool:      # pragma: no cover - trivial
        return True

    def process(self, ctx: CognitiveContext) -> CognitiveContext:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# helpers over the word-meaning state
# --------------------------------------------------------------------------
def _genus(wm: dict, word: str) -> str:
    b = (wm.get("beliefs") or {}).get(word)
    return b.get("genus", "") if b else ""


def _understood(wm: dict, word: str) -> bool:
    b = (wm.get("beliefs") or {}).get(word)
    return bool(b and b.get("understood"))


def _neighbours(wm: dict, word: str) -> list[str]:
    return [w for w, _ in Counter((wm.get("contexts") or {}).get(word, {})).most_common(12)]


def _norm(v: str) -> str:
    return (v or "").strip()


def _book_verbs(events: list) -> list[str]:
    return [_norm(e.get("verb")) for e in events if _norm(e.get("verb"))]


def _compatible(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or {a, b} == {"人", "生き物"}


# --------------------------------------------------------------------------
# 1. related-memory retrieval  (semantic, not string match)
# --------------------------------------------------------------------------
class Retrieval(CognitiveModule):
    name = "retrieval"

    def process(self, ctx: CognitiveContext) -> CognitiveContext:
        c, wm = ctx.concept, ctx.wm
        if not c:
            return ctx
        g = ctx.concept_genus
        out: list[dict] = []
        b = (wm.get("beliefs") or {}).get(c)
        if b:
            out.append({"memory_id": _hid("belief", c), "kind": "belief",
                        "content": {"word": c, "genus": b.get("genus"),
                                    "confidence": b.get("confidence"),
                                    "understood": b.get("understood")},
                        "relevance": 1.0, "source": "reading-word-meaning",
                        "created": b.get("last_cycle"), "used_count": 0})
        # events of the current book that mention the concept
        for i, e in enumerate(ctx.events):
            if c in (_norm(e.get("subject")), _norm(e.get("obj"))):
                out.append({"memory_id": _hid("event", ctx.book_id, str(i)), "kind": "event",
                            "content": {"subject": e.get("subject"), "verb": e.get("verb"),
                                        "obj": e.get("obj"), "index": i,
                                        "next_verb": _norm(ctx.events[i + 1].get("verb"))
                                        if i + 1 < len(ctx.events) else ""},
                            "relevance": 0.8, "source": ctx.book_id,
                            "created": ctx.cycle, "used_count": 0})
        # neighbours that Noise understands, same or related genus
        for n in _neighbours(wm, c):
            ng = _genus(wm, n)
            if ng and _understood(wm, n):
                rel = 0.6 if _compatible(ng, g) else 0.4
                out.append({"memory_id": _hid("belief", n), "kind": "belief",
                            "content": {"word": n, "genus": ng}, "relevance": rel,
                            "source": "reading-word-meaning", "created": None, "used_count": 0})
        ctx.retrieved = sorted(out, key=lambda r: -r["relevance"])[:20]
        ctx.trace(f"retrieval: {len(ctx.retrieved)} memories for {c!r} ({g or '?'})")
        return ctx


# --------------------------------------------------------------------------
# 2. abstraction -- generalise events by genus into reusable rules
# --------------------------------------------------------------------------
def _abstract_rules(state: dict, heur_store: dict, wm: dict, cycle: int) -> None:
    """(subject_genus, verb, object_genus) triples seen in >= RULE_MIN_BOOKS
    books become rules.  Support accumulates; a genus revised in beliefs marks
    the rule for recheck; contradicting a reusable rule adds a counterexample."""
    rules = {r["rule_id"]: r for r in state.get("rules", [])}
    triple_books: dict[str, set] = defaultdict(set)
    triple_count: Counter = Counter()
    for bid, events in heur_store.items():
        for e in events:
            if e.get("provenance") != "heuristic_self":
                continue
            sg = _genus(wm, _norm(e.get("subject")))
            og = _genus(wm, _norm(e.get("obj")))
            v = _norm(e.get("verb"))
            if not (sg and _informative_verb(v)):
                continue
            key = f"{sg}|{v}|{og}"
            triple_books[key].add(bid)
            triple_count[key] += 1
    for key, books in triple_books.items():
        if len(books) < RULE_MIN_BOOKS or triple_count[key] < RULE_MIN_SUPPORT:
            continue
        sg, v, og = key.split("|")
        rid = _hid("rule", key)
        r = rules.get(rid)
        support = triple_count[key]
        nb = len(books)
        status = ("reusable" if nb >= RULE_REUSABLE_BOOKS and support >= RULE_REUSABLE_SUPPORT
                  else "tentative")
        if r is None:
            rules[rid] = {"rule_id": rid, "form": f"「{sg}」が{v}" + (f"（「{og}」を）" if og else ""),
                          "subject_genus": sg, "verb": v, "object_genus": og,
                          "support": support, "books": sorted(books), "confidence": round((support) / (support + 2), 3),
                          "counterexamples": [], "status": status,
                          "first_cycle": cycle, "last_cycle": cycle}
        else:
            r["support"], r["books"] = support, sorted(books)
            r["confidence"] = round(support / (support + 2 + len(r.get("counterexamples", []))), 3)
            r["last_cycle"] = cycle
            if r["status"] != "weakened":
                r["status"] = status
    state["rules"] = sorted(rules.values(),
                            key=lambda r: (r["status"] != "reusable", -r["support"]))[:RULES_CAP]


class Abstraction(CognitiveModule):
    name = "abstraction"

    def process(self, ctx: CognitiveContext) -> CognitiveContext:
        ctx.trace(f"abstraction: {sum(1 for r in ctx.wm.get('_rules_view', []))} (see state)")
        return ctx


# --------------------------------------------------------------------------
# 5. problem generation -- only problems Noise can auto-grade from its data
# --------------------------------------------------------------------------
def _options(gold: str, pool: list[str], salt: str, k: int = 3) -> list[str]:
    opts = [gold] + [p for p in pool if p and p != gold][: k - 1]
    opts = list(dict.fromkeys(opts))
    opts.sort(key=lambda o: hashlib.md5(f"{salt}{o}".encode()).hexdigest())
    return opts


# which problem types measure composition (probe-eligible) vs plain recall
COMPOSITION_TYPES = ("odd_one_out", "common_property", "property_transfer")
_DOMINANT = "生き物"


def _real_word(w: str) -> bool:
    try:
        from japanese_word_meaning_v1 import _is_wordlike
        if not _is_wordlike(w):
            return False
    except Exception:
        pass
    return not any(s in w for s in ("しな", "して", "った", "ある", "いる", "れる", "など",
                                    "こと", "もの", "とき", "ため", "よう", "実際"))


def _genus_pool(wm: dict, min_conf: float = 0.55) -> "dict[str, list[str]]":
    pool: dict[str, list[str]] = defaultdict(list)
    for w, b in (wm.get("beliefs") or {}).items():
        if b.get("understood") and b.get("genus") and float(b.get("confidence", 0)) >= min_conf \
                and 2 <= len(w) <= 6 and _real_word(w):
            pool[_canon_class(b["genus"])].append(w)
    for g in pool:
        pool[g].sort(key=lambda w: hashlib.md5(w.encode()).hexdigest())
    return pool


def composition_problems(wm: dict, cycle: int, limit: int = 40) -> list[dict]:
    """Genus-combination problems drawn from the whole understood vocabulary
    (not one concept's neighbours) -- these need TWO+ beliefs combined and a
    frequency guess is wrong.  Used to build the frozen probe."""
    pool = _genus_pool(wm)
    probs: list[dict] = []
    minority = [g for g in pool if g != _DOMINANT and len(pool[g]) >= 2]
    # odd_one_out: two from a minority genus + one dominant  (gold = dominant one)
    for g in minority:
        ws = pool[g]
        for i in range(0, len(ws) - 1, 2):
            if len(probs) >= limit:
                break
            a, b = ws[i], ws[i + 1]
            odd = pool[_DOMINANT][(i // 2) % len(pool[_DOMINANT])] if pool.get(_DOMINANT) else ""
            if not odd or odd in (a, b):
                continue
            probs.append({"pid": _hid("cp", g, a, b), "level": 2, "type": "odd_one_out",
                          "concept": a, "prompt": f"「{a}」「{b}」「{odd}」のうち、種類がちがうのは？",
                          "options": _options(odd, [a, b], g + a + b), "gold": odd, "grade": "exact"})
    # common_property: two words of the same minority genus  (gold = that genus)
    for g in minority:
        ws = pool[g]
        for i in range(0, len(ws) - 1, 2):
            if len(probs) >= limit:
                break
            a, b = ws[i], ws[i + 1]
            probs.append({"pid": _hid("cc", g, a, b), "level": 3, "type": "common_property",
                          "concept": a, "other": b,
                          "prompt": f"「{a}」と「{b}」に共通するのは？",
                          "options": _options(g, list(COARSE), g + a + b + "c"),
                          "gold": g, "grade": "coarse"})
    probs.sort(key=lambda p: hashlib.md5(p["pid"].encode()).hexdigest())
    return probs[:limit]


def generate_problems(ctx: CognitiveContext, rules: list[dict], limit: int) -> list[dict]:
    wm, c, g = ctx.wm, ctx.concept, ctx.concept_genus
    probs: list[dict] = []
    verbs = _book_verbs(ctx.events)
    nb = _neighbours(wm, c)

    # L1 (recall, live practice only) -- what did the concept do next?
    for i, e in enumerate(ctx.events[:-1]):
        if _norm(e.get("subject")) == c and _informative_verb(_norm(ctx.events[i + 1].get("verb"))):
            gold = _norm(ctx.events[i + 1].get("verb"))
            probs.append({"pid": _hid("p", ctx.book_id, "evrec", str(i)), "level": 1,
                          "type": "event_recall", "concept": c,
                          "prompt": f"「{ctx.shelf.get(ctx.book_id, {}).get('title', ctx.book_id)}」で"
                                    f"「{c}」が「{_norm(e.get('verb'))}」の次にしたことは？",
                          "options": _options(gold, verbs, ctx.book_id + "evrec" + str(i)),
                          "gold": gold, "grade": "exact"})
            break

    # L1 (recall) -- genus recall
    if g and _understood(wm, c):
        probs.append({"pid": _hid("p", c, "gen"), "level": 1, "type": "genus_recall", "concept": c,
                      "prompt": f"「{c}」は次のどれ？",
                      "options": _options(g, list(COARSE), c + "gen"), "gold": g, "grade": "coarse"})

    # L2 (composition) -- odd one out: classify 3 words, name the mismatch
    same = [w for w in nb if _genus(wm, w) == g and _understood(wm, w) and w != c][:2]
    diff = next((w for w in nb if _genus(wm, w) and not _compatible(_genus(wm, w), g)
                 and _understood(wm, w)), "")
    if g and len(same) == 2 and diff:
        probs.append({"pid": _hid("p", c, "odd"), "level": 2, "type": "odd_one_out",
                      "concept": c, "members": [c] + same, "odd": diff,
                      "prompt": f"「{c}」「{same[0]}」「{same[1]}」「{diff}」のうち、仲間はずれは？",
                      "options": _options(diff, [c] + same, c + "odd"), "gold": diff, "grade": "exact"})

    # L3 (composition) -- common property of two understood words
    for n in nb:
        ng = _genus(wm, n)
        if n != c and ng and g and _understood(wm, n) and _compatible(ng, g):
            common = g if g == ng else "生き物"
            probs.append({"pid": _hid("p", c, n, "common"), "level": 3, "type": "common_property",
                          "concept": c, "other": n,
                          "prompt": f"「{c}」と「{n}」に共通するのは？",
                          "options": _options(common, list(COARSE), c + n + "common"),
                          "gold": common, "grade": "coarse"})
            break

    # L4 (composition) -- transfer: from rules learned on OTHER books, predict
    # what the concept actually does in THIS book (its events are held out for
    # this problem).  Gold is a real observed verb, not the rule's verb.
    actual_verbs = [_norm(e.get("verb")) for e in ctx.events
                    if _norm(e.get("subject")) == c and _informative_verb(_norm(e.get("verb")))]
    if actual_verbs:
        gold = Counter(actual_verbs).most_common(1)[0][0]
        xfer = [x for x in rules if x["status"] == "reusable" and not x["object_genus"]
                and _compatible(x["subject_genus"], g) and _informative_verb(x["verb"])
                and ctx.book_id not in x["books"]]
        if xfer:
            r = xfer[0]
            distract = [x["verb"] for x in rules[:8] if x["verb"] != gold] + verbs
            probs.append({"pid": _hid("p", c, ctx.book_id, "xfer"), "level": 4,
                          "type": "property_transfer", "concept": c, "rule_id": r["rule_id"],
                          "prompt": f"別の話から学んだ規則から予想して、「{c}」はこの話で何をする？",
                          "options": _options(gold, distract, c + ctx.book_id + "xfer"),
                          "gold": gold, "grade": "exact"})

    seen = set()
    uniq = []
    for p in probs:
        if p["pid"] in seen or not p["options"] or p["gold"] not in p["options"]:
            continue
        seen.add(p["pid"])
        uniq.append(p)
    return uniq[:limit]


# --------------------------------------------------------------------------
# 6. reasoning -- explicit retrieve / match / compose, step by step
# --------------------------------------------------------------------------
def solve(problem: dict, wm: dict, rules: list[dict], corrections: list[str],
          heur_store: dict | None = None, compose: bool = False,
          deliberate: bool = False) -> dict:
    steps: list[str] = []
    t, concept = problem["type"], problem.get("concept", "")
    rules_by_id = {r["rule_id"]: r for r in rules}
    answer, conf = "わからない", 0.0
    ab = not compose      # compose mode: derive genus from neighbours, not lookup
    hs = heur_store or {}

    if corrections:
        steps.append(f"past correction: {corrections[0]}")

    def classify(word: str) -> str:
        g = _infer_genus(wm, word, allow_belief=ab)
        if not deliberate:
            return g
        # deliberate: hypotheses -> counter-evidence -> best survivor.  Use it
        # when there is no direct answer, or when the direct answer is itself
        # contradicted by the evidence (don't trust a wrong belief).
        if g and not _contradicted(g, word, wm, hs, steps):
            return g
        dg, _dc = deliberate_genus(word, wm, hs, rules, steps)
        return dg or g

    if t == "genus_recall":
        b = (wm.get("beliefs") or {}).get(concept, {})
        steps.append(f"lookup belief[{concept}] -> {b.get('genus') or '?'} (understood={b.get('understood')})")
        if b.get("understood"):
            answer, conf = b.get("genus", ""), round(float(b.get("confidence", 0.5)), 3)
        if deliberate and (not b.get("understood") or _contradicted(answer, concept, wm, hs, steps)):
            g, gc = deliberate_genus(concept, wm, hs, rules, steps)
            answer, conf = (g, gc) if g else ("わからない", 0.0)

    elif t == "odd_one_out":
        # classify every option, then name the one whose genus is the minority
        cls = {}
        for opt in problem["options"]:
            og = classify(opt)
            cls[opt] = og
            steps.append(f"classify {opt} -> {og or '?'}")
        buckets: dict[str, list[str]] = defaultdict(list)
        for opt, og in cls.items():
            if og:
                buckets[_canon_class(og)].append(opt)
        if len(buckets) >= 2:
            minority = min(buckets.values(), key=len)
            if len(minority) == 1:
                answer, conf = minority[0], 0.6
                steps.append(f"minority class -> {answer}")

    elif t == "common_property":
        a, bword = concept, problem.get("other", "")
        ga, gb = classify(a), classify(bword)
        steps.append(f"genus[{a}]={ga or '?'}  genus[{bword}]={gb or '?'}")
        if ga and gb and _compatible(ga, gb):
            answer = ga if ga == gb else "生き物"
            conf = 0.55
            steps.append(f"shared genus -> {answer}")

    elif t == "event_recall":
        ev = (heur_store or {}).get(problem.get("book_id", ""), [])
        for i, e in enumerate(ev[:-1]):
            if _norm(e.get("subject")) == concept:
                nxt = _norm(ev[i + 1].get("verb"))
                steps.append(f"found {concept} at event {i}; next verb = {nxt or '?'}")
                if nxt:
                    answer, conf = nxt, 0.5
                break
        if answer == "わからない":
            steps.append("concept not found in stored events")

    elif t == "property_transfer":
        r = rules_by_id.get(problem.get("rule_id"))
        cg = _infer_genus(wm, concept)
        if r:
            steps.append(f"rule {r['rule_id']} : {r['form']} (support {r['support']}, {r['status']})")
            steps.append(f"genus of {concept} = {cg or '?'}; rule subject = {r['subject_genus']}")
            if _compatible(cg, r["subject_genus"]):
                answer = r["verb"]
                conf = round(r["confidence"] * 0.7, 3)
            else:
                steps.append("genus mismatch -> rule does not apply")
        else:
            steps.append("rule missing from state")

    if len(steps) > MAX_REASONING_STEPS:
        steps = steps[:MAX_REASONING_STEPS] + ["... (truncated)"]
    return {"answer": answer, "confidence": conf, "steps": steps}


def _canon_class(g: str) -> str:
    return "生き物" if g == "人" else g


def _contradicted(genus: str, word: str, wm: dict, heur_store: dict, steps: list[str]) -> bool:
    if not genus:
        return False
    ce = seek_counterevidence({"genus": _canon_class(genus), "confidence": 0.6}, word, wm, heur_store)
    if ce["counters"]:
        steps.append(f"{word}={genus} contradicted: {ce['counters'][0]}")
        return True
    return False


def _infer_genus(wm: dict, word: str, allow_belief: bool = True) -> str:
    """Belief genus if Noise understands the word (and `allow_belief`);
    otherwise compose it from the genera of its understood neighbours.  The
    probe passes allow_belief=False so it measures composition, not lookup."""
    b = (wm.get("beliefs") or {}).get(word)
    if allow_belief and b and b.get("understood"):
        return b.get("genus", "")
    votes = Counter()
    for n in _neighbours(wm, word):
        nb = (wm.get("beliefs") or {}).get(n)
        if nb and nb.get("understood") and nb.get("genus"):
            votes[_canon_class(nb["genus"])] += 1
    if votes:
        return votes.most_common(1)[0][0]
    return b.get("genus", "") if (b and allow_belief) else ""


# --------------------------------------------------------------------------
# 3. hypotheses  +  4. counter-evidence  (used only when direct reasoning stalls)
# --------------------------------------------------------------------------
# a morphological hint is LOW evidence -- a suffix, not authority
_MORPH_HINTS = (
    (("器", "具", "機", "刀", "筒", "鏡", "杖", "笛", "鈴", "鍬", "車", "船", "服", "衣",
      "帽子", "靴", "傘", "鉄砲", "貨", "金"), "道具"),
    (("山", "川", "海", "湖", "島", "森", "林", "村", "町", "国", "橋", "寺", "宮", "門",
      "庭", "畑", "野原", "坂", "道", "院", "駅", "校"), "場所"),
    (("鳥", "犬", "猫", "狐", "狸", "熊", "鹿", "兎", "馬", "牛", "羊", "虫", "魚", "猿",
      "狼", "獅子", "象", "蛇", "蛙", "亀"), "生き物"),
    (("さん", "君", "氏", "王", "姫", "僧", "侍", "師", "翁", "嬢", "娘", "息子",
      "父", "母", "兄", "姉", "夫", "妻", "者"), "人"),
    (("料理", "菓子", "飯", "汁", "酒", "茶", "パン", "肉", "果"), "食べ物"),
)
# perception / speech / feeling / eating -- needs a mind.  NOT motion (走/歩/飛):
# a train moves too.  NOT 鳴 (鳴る = an object ringing).
_ANIMATE_HINT = ("見", "みる", "聞", "きく", "言", "いう", "話", "はなす", "食べ", "たべ",
                 "飲", "のむ", "泣", "なく", "笑", "わら", "思", "おもう", "考", "かんが",
                 "寝", "ねる", "答", "こたえ", "逃", "にげ", "遊", "あそ", "怒", "おこ")
_HANDLE_HINT = ("つく", "作", "持", "使", "つか", "投げ", "取", "とる", "買", "売",
                "こわ", "壊", "割", "切", "ひらく", "開", "しめ", "運", "置", "拾",
                "引", "掴", "にぎ", "握", "かつ", "担", "はく", "着", "かぶ")


def _handled(verb: str) -> bool:
    v = _norm(verb)
    return any(h in v for h in _HANDLE_HINT)


def _morph_genus(word: str) -> str:
    for keys, g in _MORPH_HINTS:
        if any(k in word for k in keys):
            return g
    return ""


def generate_hypotheses(word: str, wm: dict, heur_store: dict, rules: list[dict]) -> list[dict]:
    """Several independent guesses at `word`'s coarse class -- never fix on one."""
    hyps: list[dict] = []
    seen = set()

    def add(genus, basis, conf, check):
        g = _canon_class(genus)
        if g in COARSE and g not in seen:
            seen.add(g)
            hyps.append({"genus": g, "basis": basis, "confidence": round(conf, 3), "check": check})

    b = (wm.get("beliefs") or {}).get(word) or {}
    if b.get("understood"):
        add(b.get("genus", ""), ["belief"], min(0.7, float(b.get("confidence", 0.5))),
            "does the word act like this class in the stories?")
    nv = _infer_genus(wm, word, allow_belief=False)
    if nv:
        add(nv, ["understood neighbours"], 0.45, "do most neighbours really share this class?")
    mg = _morph_genus(word)
    if mg:
        add(mg, ["morphology"], 0.3, "is the suffix actually meaningful here?")
    # rule-based: word is the object of a handling verb -> a thing
    for bid, evs in list(heur_store.items())[:120]:
        for e in evs:
            if _norm(e.get("obj")) == word and _handled(_norm(e.get("verb"))):
                add("道具", [f"handled in {bid}"], 0.35, "is it ever an agent too?")
                break
        if "道具" in seen:
            break
    return hyps


def _usage(word: str, wm: dict, heur_store: dict) -> tuple[int, int, bool]:
    """(times as subject, times as object, ever did an animate action).
    Prefer word_meaning's accumulated `profiles`; fall back to the events."""
    prof = (wm.get("profiles") or {}).get(word)
    if prof:
        sv = prof.get("subj_verbs", {})
        animate = any(any(h in v for h in _ANIMATE_HINT) for v in sv)
        return int(prof.get("subj", 0)), int(prof.get("obj", 0)), animate
    subj = obj = 0
    animate = False
    for evs in list(heur_store.values())[:180]:
        for e in evs:
            v = _norm(e.get("verb"))
            if _norm(e.get("subject")) == word:
                subj += 1
                if any(h in v for h in _ANIMATE_HINT):
                    animate = True
            if _norm(e.get("obj")) == word:
                obj += 1
    return subj, obj, animate


def seek_counterevidence(hyp: dict, word: str, wm: dict, heur_store: dict) -> dict:
    """Actively try to break the hypothesis -- not confirm it (user #4)."""
    g, counters = hyp["genus"], []
    subj, obj, animate = _usage(word, wm, heur_store)
    if g in ("生き物", "人"):
        # a creature acts; if the word is only ever acted upon, that is evidence against
        if obj >= 3 and subj == 0:
            counters.append(f"{word} は常に対象で、一度も動作主でない（{obj}回）")
        elif subj >= 2 and not animate:
            counters.append(f"{word} は主語だが生き物的な行動をしない")
    if g in ("道具", "場所", "食べ物", "自然物"):
        if animate:
            counters.append(f"{word} は生き物的な行動をする")
    nb = Counter(_canon_class(_genus(wm, n)) for n in _neighbours(wm, word)
                 if _understood(wm, n) and _genus(wm, n))
    if nb and len(nb) > 1:                       # a mixed neighbourhood, not a clean one
        top, k = nb.most_common(1)[0]
        if top != g and k >= 4 and k >= 2 * nb.get(g, 0):
            counters.append(f"近傍語の多数（{k}）は「{top}」")
    surviving = round(hyp["confidence"] * (0.35 ** len(counters)), 3)
    return {"counters": counters, "surviving_confidence": surviving}


class HypothesisGen(CognitiveModule):
    """#3 -- several independent guesses for the focus concept's class."""
    name = "hypothesis"

    def process(self, ctx: CognitiveContext) -> CognitiveContext:
        if ctx.concept:
            hs = generate_hypotheses(ctx.concept, ctx.wm, ctx.heur_store, [])
            ctx.trace(f"hypotheses({ctx.concept}): "
                      + ", ".join(f"{h['genus']}@{h['confidence']}" for h in hs))
        return ctx


class CounterSearch(CognitiveModule):
    """#4 -- try to break each hypothesis, not confirm it."""
    name = "counter"

    def process(self, ctx: CognitiveContext) -> CognitiveContext:
        if ctx.concept:
            for h in generate_hypotheses(ctx.concept, ctx.wm, ctx.heur_store, []):
                ce = seek_counterevidence(h, ctx.concept, ctx.wm, ctx.heur_store)
                if ce["counters"]:
                    ctx.trace(f"counter {ctx.concept}={h['genus']}: {ce['counters'][0]}")
        return ctx


def deliberate_genus(word: str, wm: dict, heur_store: dict, rules: list[dict],
                     steps: list[str]) -> tuple[str, float]:
    """Generate hypotheses, break each against the evidence, take the best
    survivor.  Returns ('', 0.0) if nothing survives -- an honest 'わからない'."""
    hyps = generate_hypotheses(word, wm, heur_store, rules)
    if not hyps:
        steps.append(f"deliberate({word}): no hypothesis")
        return "", 0.0
    scored = []
    for h in hyps:
        ce = seek_counterevidence(h, word, wm, heur_store)
        steps.append(f"H {word}={h['genus']} (conf {h['confidence']}, {','.join(h['basis'])}) "
                     f"-> counters {len(ce['counters'])} -> {ce['surviving_confidence']}")
        scored.append((ce["surviving_confidence"], len(ce["counters"]), h["genus"]))
    scored.sort(reverse=True)
    best_conf, best_counters, best_g = scored[0]
    if best_conf < 0.12 or best_counters:
        steps.append("no hypothesis survives the counter-evidence -> unresolved")
        return "", 0.0
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.05 and scored[0][2] != scored[1][2]:
        steps.append("two hypotheses survive equally -> unresolved")
        return "", 0.0
    return best_g, best_conf


# --------------------------------------------------------------------------
# 7. self-evaluation of the reasoning (not correctness -- that is graded)
# --------------------------------------------------------------------------
def self_evaluate(problem: dict, solution: dict, retrieved: list[dict], wm: dict) -> dict:
    used_knowledge = any("lookup" in s or "rule" in s or "found" in s or "belief[" in s
                         for s in solution["steps"])
    in_options = solution["answer"] in problem["options"]
    # contradiction: the answer names a coarse class that disagrees with the concept's belief
    contradiction = False
    if problem["grade"] == "coarse":
        bg = ((wm.get("beliefs") or {}).get(problem.get("concept", ""), {})).get("genus", "")
        if bg and solution["answer"] in COARSE and not _compatible(bg, solution["answer"]) \
                and problem["type"] != "common_property":
            contradiction = True
    calibrated = (solution["confidence"] > 0) == (solution["answer"] != "わからない")
    notes = []
    if not used_knowledge:
        notes.append("no knowledge retrieved/used")
    if contradiction:
        notes.append("answer contradicts the concept's belief")
    if not in_options and solution["answer"] != "わからない":
        notes.append("answer outside the option set")
    return {"sufficient_basis": used_knowledge, "in_option_set": in_options,
            "contradiction": contradiction, "confidence_calibrated": calibrated,
            "notes": notes}


# --------------------------------------------------------------------------
# 8. experience memory -- Situation / Thought / Action / Result / Eval / Correction
# --------------------------------------------------------------------------
def _correction_for(problem: dict, solution: dict, correct: bool, evaluation: dict) -> str:
    if correct:
        return f"reasoning path for {problem['type']} works; keep using belief/rule lookup"
    if not evaluation["sufficient_basis"]:
        return "retrieve wider next time: concept genus + understood neighbours"
    if evaluation["contradiction"]:
        return f"belief for {problem.get('concept')} may be wrong -- recheck on next reading"
    if problem["type"] in ("property_transfer",):
        return f"rule {problem.get('rule_id')} unreliable for {problem.get('concept')} -- add counterexample"
    return f"got {solution['answer']!r}, wanted {problem['gold']!r} -- prefer direct belief lookup"


def record_experience(state: dict, problem: dict, solution: dict, correct: bool,
                      evaluation: dict, cycle: int) -> dict:
    concept_genus = ((state.get("_wm_beliefs") or {}).get(problem.get("concept", ""), {})).get("genus", "")
    sig = f"{problem['type']}:{problem.get('concept', '')}"
    prior = [e for e in state.get("experiences", []) if e.get("signature") == sig]
    repeated = bool(prior) and not correct and any("wrong" in e.get("result", "") for e in prior[-3:])
    rec = {"exp_id": _hid("exp", problem["pid"], str(cycle)), "cycle": cycle,
           "signature": sig, "problem_type": problem["type"], "level": problem["level"],
           "concept": problem.get("concept", ""), "concept_genus": concept_genus,
           "situation": problem["prompt"],
           "thought": solution["steps"],
           "action": [s for s in solution["steps"] if "rule" in s or "belief[" in s or "lookup" in s
                      or "classify" in s or "found" in s],
           "result": "correct" if correct else f"wrong: got {solution['answer']!r} want {problem['gold']!r}",
           "repeated_failure": repeated,
           "evaluation": evaluation["notes"] or ["reasoning ok"],
           "correction": _correction_for(problem, solution, correct, evaluation)}
    exps = state.setdefault("experiences", [])
    exps.append(rec)
    state["experiences"] = exps[-EXPERIENCE_CAP:]
    idx = state.setdefault("corrections_index", {})
    key = f"{problem['type']}:{concept_genus or '?'}"
    lst = idx.setdefault(key, [])
    if rec["correction"] not in lst:
        lst.insert(0, rec["correction"])
    idx[key] = lst[:5]
    # a failed rule application is a counterexample (invariant 8)
    if not correct and problem["type"] in ("property_transfer",) and problem.get("rule_id"):
        for r in state.get("rules", []):
            if r["rule_id"] == problem["rule_id"]:
                ce = {"cycle": cycle, "concept": problem.get("concept"),
                      "predicted": solution["answer"], "wanted": problem["gold"]}
                r.setdefault("counterexamples", []).append(ce)
                r["counterexamples"] = r["counterexamples"][-8:]
                if len(r["counterexamples"]) >= 3:
                    r["status"] = "weakened"
                r["confidence"] = round(r["support"] / (r["support"] + 2 + len(r["counterexamples"])), 3)
    return rec


def corrections_for(state: dict, problem_type: str, concept_genus: str) -> list[str]:
    idx = state.get("corrections_index", {})
    return idx.get(f"{problem_type}:{concept_genus or '?'}", []) or idx.get(f"{problem_type}:?", [])


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def _pick_concept(wm: dict, events: list, previous: dict, cycle: int) -> str:
    """Rotate through understood entities of the book just read; fall back to
    the most-read understood entity overall."""
    beliefs = wm.get("beliefs") or {}
    in_book = [w for w in {_norm(e.get("subject")) for e in events} | {_norm(e.get("obj")) for e in events}
               if w and beliefs.get(w, {}).get("understood")]
    if in_book:
        return sorted(in_book)[cycle % len(in_book)]
    understood = [w for w, b in beliefs.items() if b.get("understood")]
    ent = wm.get("entities") or {}
    understood.sort(key=lambda w: -ent.get(w, 0))
    return understood[cycle % len(understood)] if understood else ""


def run_cognitive_cycle(cycle: int, wm_state: dict, heur_store: dict, shelf: dict,
                        just_read: str, previous: dict | None) -> dict:
    state = dict(previous or {})
    state.setdefault("version", VERSION)
    if state.get("version") != VERSION:
        state = {"version": VERSION}
    state.setdefault("rules", [])
    state.setdefault("experiences", [])
    state.setdefault("corrections_index", {})
    state.setdefault("loop_history", [])
    state["_wm_beliefs"] = wm_state.get("beliefs") or {}

    # 2. abstraction always runs over the full read corpus (cheap, capped)
    _abstract_rules(state, heur_store, wm_state, cycle)

    events = heur_store.get(just_read, []) if just_read else []
    events = [e for e in events if e.get("provenance") == "heuristic_self"]
    concept = _pick_concept(wm_state, events, previous or {}, cycle)

    ctx = CognitiveContext(cycle=cycle, concept=concept, concept_genus=_genus(wm_state, concept),
                           events=events, book_id=just_read or "", wm=wm_state,
                           heur_store=heur_store, shelf=shelf)

    solved_summary = {"total": 0, "correct": 0, "by_level": {}}
    if concept:
        Retrieval().process(ctx)
        ctx.problems = generate_problems(ctx, state["rules"], PROBLEMS_PER_CYCLE)
        for p in ctx.problems:
            p.setdefault("book_id", just_read or "")
            corr = corrections_for(state, p["type"], ctx.concept_genus)
            sol = solve(p, wm_state, state["rules"], corr, heur_store, deliberate=True)
            correct = _grade(p, sol["answer"])
            ev = self_evaluate(p, sol, ctx.retrieved, wm_state)
            record_experience(state, p, sol, correct, ev, cycle)
            ctx.solved.append({"pid": p["pid"], "level": p["level"], "type": p["type"],
                               "answer": sol["answer"], "gold": p["gold"], "correct": correct,
                               "confidence": sol["confidence"], "evaluation": ev["notes"]})
            solved_summary["total"] += 1
            solved_summary["correct"] += int(correct)
            lv = solved_summary["by_level"].setdefault(str(p["level"]), {"n": 0, "ok": 0})
            lv["n"] += 1
            lv["ok"] += int(correct)

    hist = state["loop_history"]
    hist.append({"cycle": cycle, "concept": concept, "n_rules": len(state["rules"]),
                 "reusable_rules": sum(1 for r in state["rules"] if r["status"] == "reusable"),
                 **solved_summary})
    state["loop_history"] = hist[-HISTORY_CAP:]

    state.pop("_wm_beliefs", None)
    reusable = sum(1 for r in state["rules"] if r["status"] == "reusable")
    recent = [h for h in state["loop_history"][-30:] if h["total"]]
    live_rate = round(sum(h["correct"] for h in recent) / max(1, sum(h["total"] for h in recent)), 3)
    recent_exp = state["experiences"][-200:]
    attempts = [e for e in recent_exp if "wrong" in e["result"] or e["result"] == "correct"]
    repeated = sum(1 for e in recent_exp if e.get("repeated_failure"))
    repeat_rate = round(repeated / max(1, sum(1 for e in recent_exp if "wrong" in e["result"])), 3)
    return {**state, "status": "ran" if concept else "no_concept",
            "cycle": cycle, "concept": concept,
            "rules_total": len(state["rules"]), "rules_reusable": reusable,
            "experiences_total": len(state["experiences"]),
            "problems_this_cycle": solved_summary["total"],
            "correct_this_cycle": solved_summary["correct"],
            "live_solve_rate": live_rate,
            "repeated_failure_rate": repeat_rate,
            "by_level": solved_summary["by_level"],
            "sample_experience": state["experiences"][-1] if state["experiences"] else None,
            "note": "capability is measured by capability_probe_v1 on a frozen set, "
                    "not by rule count (ARCHITECTURE invariant 16)"}


def _grade(problem: dict, answer: str) -> bool:
    if answer == "わからない" or not answer:
        return False
    if problem["grade"] == "coarse":
        return _compatible(answer, problem["gold"])
    return answer == problem["gold"]
