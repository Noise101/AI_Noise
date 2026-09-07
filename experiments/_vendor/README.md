# Vendored: Janome 0.5.0

A pure-Python Japanese morphological analyser (MeCab IPADIC, no C extension).
Used **only** by `morphology_teacher.py` as a disclosed, non-authoritative
reference for the developmental reading loop -- its output is a proposal with
evidence score 0, exactly like the optional local-LLM helpers.  It never feeds
the frozen benchmarks, the character RNN, or the `japanese_boundaries_v18`
autonomous-discovery claim (see `ARCHITECTURE.md`, "Optional local-model /
reference boundary").

## Provenance

- File: `Janome-0.5.0-py2.py3-none-any.whl`
- Source: https://files.pythonhosted.org/packages/73/7d/70f4069f4bbf0fca023e82a1fbbade6f5216365d4fe259fee1950723eca5/Janome-0.5.0-py2.py3-none-any.whl
- sha256: `d098670394a77881ce2f6b7d696c0ea5ff74c0c8cf74a8a882159ec82c0e6dc7`
- License: Apache-2.0 (`Janome-LICENSE.txt`, `Janome-NOTICE.txt`)
- Bundled dictionary: mecab-ipadic-2.7.0-20070801 (also Apache-2.0 per NOTICE)

## Why vendored, not pip-installed

This environment has no `pip` / `ensurepip`.  The wheel is committed so the
teacher works offline; `morphology_teacher.py` unpacks it to
`_vendor/janome-runtime/` (gitignored) on first use because the dictionary is
`mmap`-ed and cannot be imported from inside the zip.  A system install of
`janome` or `mecab` is preferred when present and takes precedence.
