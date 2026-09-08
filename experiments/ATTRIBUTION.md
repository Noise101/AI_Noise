# Third-party corpus attribution

AI_Noise learns to read from public text it fetches read-only. Sources:

## Aozora Bunko (aozora.gr.jp)
Public-domain Japanese literature. Works are out of copyright in Japan.
Fetched via `japanese_corpus_v1.fetch_aozora`.

## Japanese Wikisource (ja.wikisource.org)
`イソップ童話集` and a handful of folktales, under the Wikimedia terms of use
(text is CC BY-SA / public domain depending on the page). Fetched via
`japanese_corpus_v1.fetch`.

## Tatoeba (tatoeba.org)
Example sentences contributed by the Tatoeba community, released under
**CC-BY 2.0 FR** (https://creativecommons.org/licenses/by/2.0/fr/). Bulk export
`jpn_sentences.tsv.bz2` from https://downloads.tatoeba.org/. Fetched via
`japanese_corpus_v1.tatoeba_readers`; used to build graded "readers" (bundles of
short sentences) that build vocabulary and character-model training text. Each
shelf entry sourced from Tatoeba carries `source: "tatoeba"` and
`license: "CC-BY-2.0-FR"`. Attribution string:
`japanese_corpus_v1.TATOEBA_ATTRIBUTION`.

Use here is read-only and internal to the learning loop; sentences are not
redistributed.
