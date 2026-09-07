"""Optional Japanese morphological analyser, used as a *disclosed reference*.

Like the local-LLM helpers (`local_candidate_helper`, `reading_llm_v1`), this is
never authority.  Every analysis it returns carries `evidence_score = 0` and
must be corroborated (a second source, an observable story feature, the
heuristic parser agreeing) before it can change a belief.  It never feeds the
frozen benchmarks, the character RNN, or the `japanese_boundaries_v18`
autonomous-discovery path -- when no analyser is installed the reading loop runs
exactly as before on its own particle heuristics.

Backends, in order of preference: a system `fugashi`/`MeCab`, a system `janome`,
then the wheel vendored under `_vendor/` (unpacked once because its dictionary is
mmap-ed and cannot be imported from inside the zip).  `AI_NOISE_NO_MORPHOLOGY=1`
forces the null backend (tests that must stay analyser-independent set it).
"""

from __future__ import annotations

import os
import sys
import zipfile
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_WHEEL = os.path.join(_HERE, "_vendor", "Janome-0.5.0-py2.py3-none-any.whl")
_RUNTIME = os.path.join(_HERE, "_vendor", "janome-runtime")

# case particles the reading loop cares about
_CASE = {"が", "を", "に", "へ", "と", "で", "から", "より", "まで"}
_TOPIC = {"は", "も"}


@dataclass
class Morpheme:
    surface: str
    pos: str            # 名詞 / 動詞 / 助詞 / 助動詞 / 副詞 / 記号 / ...
    pos_detail: str      # 格助詞 / 係助詞 / 自立 / 非自立 / 一般 / ...
    base: str            # dictionary form (== surface when the analyser has none)

    @property
    def is_verb(self) -> bool:
        return self.pos in ("動詞", "形容詞")

    @property
    def is_case_particle(self) -> bool:
        return self.pos == "助詞" and self.surface in _CASE

    @property
    def is_topic_particle(self) -> bool:
        return self.pos == "助詞" and self.surface in _TOPIC


@dataclass
class MorphAnalysis:
    """A proposal from a reference analyser -- not evidence."""

    source: str
    morphemes: list[Morpheme]
    evidence_score: float = 0.0
    verified: bool = False

    def dictionary_form(self, surface_run: str) -> str | None:
        """Base form of the first verb/adjective that starts `surface_run`."""
        acc = ""
        for m in self.morphemes:
            if not acc and not surface_run.startswith(m.surface):
                continue
            acc += m.surface
            if m.is_verb:
                return m.base
            if not surface_run.startswith(acc):
                return None
        return None

    def particle_after(self, noun: str) -> str | None:
        """The particle immediately following the last morpheme whose surface
        ends `noun` (は/が/を/... as the analyser tagged it)."""
        prev_ends_noun = False
        for m in self.morphemes:
            if prev_ends_noun and m.pos == "助詞":
                return m.surface
            prev_ends_noun = m.pos in ("名詞", "代名詞") and (
                m.surface.endswith(noun) or noun.endswith(m.surface))
        return None


# --- backends ----------------------------------------------------------------
def _unpack_vendored_janome() -> bool:
    if not os.path.exists(_WHEEL):
        return False
    marker = os.path.join(_RUNTIME, "janome", "tokenizer.py")
    if not os.path.exists(marker):
        try:
            os.makedirs(_RUNTIME, exist_ok=True)
            with zipfile.ZipFile(_WHEEL) as zf:
                zf.extractall(_RUNTIME, members=[n for n in zf.namelist()
                                                if n.startswith("janome/")])
        except Exception:
            return False
    if _RUNTIME not in sys.path:
        sys.path.append(_RUNTIME)
    return os.path.exists(marker)


class _JanomeBackend:
    name = "janome"

    def __init__(self) -> None:
        self._tk = None

    def available(self) -> bool:
        try:
            import janome  # noqa: F401  (system install wins)
        except ImportError:
            if not _unpack_vendored_janome():
                return False
        try:
            from janome.tokenizer import Tokenizer  # noqa: F401
            return True
        except Exception:
            return False

    def analyse(self, sentence: str) -> "MorphAnalysis | None":
        try:
            if self._tk is None:
                from janome.tokenizer import Tokenizer
                self._tk = Tokenizer()
            morphemes = []
            for tok in self._tk.tokenize(sentence):
                parts = tok.part_of_speech.split(",")
                base = tok.base_form if tok.base_form and tok.base_form != "*" else tok.surface
                morphemes.append(Morpheme(tok.surface, parts[0],
                                          parts[1] if len(parts) > 1 else "*", base))
            return MorphAnalysis("janome", morphemes)
        except Exception:
            return None


class _FugashiBackend:
    name = "fugashi"

    def __init__(self) -> None:
        self._tagger = None

    def available(self) -> bool:
        try:
            import fugashi  # noqa: F401
            import fugashi as _f
            _f.Tagger()
            return True
        except Exception:
            return False

    def analyse(self, sentence: str) -> "MorphAnalysis | None":
        try:
            if self._tagger is None:
                import fugashi
                self._tagger = fugashi.Tagger()
            morphemes = []
            for word in self._tagger(sentence):
                feat = word.feature
                pos = getattr(feat, "pos1", None) or "*"
                pos2 = getattr(feat, "pos2", None) or "*"
                base = (getattr(feat, "orthBase", None) or getattr(feat, "lemma", None)
                        or word.surface)
                morphemes.append(Morpheme(word.surface, pos, pos2, base))
            return MorphAnalysis("fugashi", morphemes)
        except Exception:
            return None


class NullMorphologyTeacher:
    name = "none"

    def available(self) -> bool:
        return False

    def analyse(self, sentence: str) -> None:
        return None


_CACHED: "object | None" = None


def get_teacher(refresh: bool = False):
    """The best available analyser, or a null backend.  Cached per process."""
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED
    if os.environ.get("AI_NOISE_NO_MORPHOLOGY") == "1":
        _CACHED = NullMorphologyTeacher()
        return _CACHED
    for backend in (_FugashiBackend(), _JanomeBackend()):
        try:
            if backend.available():
                _CACHED = backend
                return _CACHED
        except Exception:
            continue
    _CACHED = NullMorphologyTeacher()
    return _CACHED


def analyse(sentence: str) -> "MorphAnalysis | None":
    return get_teacher().analyse(sentence)


if __name__ == "__main__":
    t = get_teacher()
    print(f"backend: {t.name}")
    for line in sys.stdin:
        a = t.analyse(line.strip())
        if not a:
            print("(no analyser)")
            continue
        for m in a.morphemes:
            print(f"  {m.surface}\t{m.pos}\t{m.pos_detail}\t{m.base}")
