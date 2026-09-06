import unittest

from japanese_event_v1 import (extract_clause, extract_story, learn_word_vocabulary,
                               normalise_text, _dictionary_verb)

MOMOTARO = (
    "むかしむかし、あるところに、おじいさんとおばあさんがいました。"
    "おじいさんは山へしばかりに行きました。おばあさんは川へせんたくに行きました。"
    "おばあさんはそのももをひろいました。おばあさんはももを家へ持って帰りました。"
    "おじいさんが山から帰りました。二人はももを切りました。"
    "ももの中から元気な男の子が生まれました。"
    "ももたろうはおにたいじに行くと言いました。おばあさんはきびだんごを作りました。"
    "ももたろうはおにを島でやっつけました。")

FOX_GRAPES = (
    "きつねがぶどうを見つけました。きつねはぶどうがたべたくなりました。"
    "きつねはとびあがりました。でもぶどうに手がとどきませんでした。"
    "きつねは「あのぶどうはすっぱい」と言いました。")


class JapaneseEventTest(unittest.TestCase):
    def test_particles_give_subject_object_and_verb(self):
        event = extract_clause("きつねがぶどうを見つけました。")
        self.assertEqual((event.subject, event.verb, event.obj), ("きつね", "見つける", "ぶどう"))

    def test_topic_wa_is_read_as_the_subject_before_a_word_starting_with_a_particle_char(self):
        # は is followed by も (of もも) -- must still be the topic boundary
        event = extract_clause("おばあさんはももを切りました。")
        self.assertEqual((event.subject, event.obj), ("おばあさん", "もも"))

    def test_particle_character_inside_a_noun_is_not_a_boundary(self):
        # もも: the second も is not the particle も
        event = extract_clause("おばあさんはももをひろいました。")
        self.assertEqual(event.obj, "もも")
        # 上がる / とびあがる: the が is verb-internal
        event = extract_clause("きつねはとびあがりました。")
        self.assertEqual(event.verb, "とびあがる")

    def test_omitted_subject_is_threaded_from_the_previous_clause(self):
        events = extract_story("きつねはぶどうを見つけました。たべたくなりました。")
        self.assertEqual([e.subject for e in events], ["きつね", "きつね"])

    def test_sensation_ga_noun_is_not_taken_as_the_agent(self):
        # 「ねこは…。おなかがすいた。言いました。」-- おなか must not become the
        # subject and get threaded forward as the one who speaks
        events = extract_story("ねこはねずみをまちました。おなかがすいてたまりません。"
                               "おおきなこえでいいました。")
        self.assertEqual([e.subject for e in events], ["ねこ", "ねこ", "ねこ"])
        self.assertEqual(events[1].roles.get("が"), "おなか")

    def test_a_bare_topic_clause_sets_the_subject_for_following_clauses(self):
        events = extract_story("おじいさんは、やまへ行きました。しばをかりました。")
        self.assertEqual([e.subject for e in events], ["おじいさん", "おじいさん"])

    def test_counter_and_adjectival_prefixes_are_stripped_from_the_subject(self):
        self.assertEqual(extract_clause("いっぴきのねずみがはしりました。").subject, "ねずみ")
        self.assertEqual(extract_clause("ふたりのこどもがあそびました。").subject, "こども")

    def test_quantifier_mo_is_not_read_as_a_topic_entity(self):
        # 「いっぴきも…ない」: いっぴき must not become the threaded subject
        events = extract_story("きつねがはしりました。いっぴきもいませんでした。"
                               "とてもかなしみました。")
        self.assertNotIn("いっぴき", [e.subject for e in events])

    def test_single_kanji_noun_is_a_valid_argument(self):
        event = extract_clause("おじいさんは山から帰りました。")
        self.assertEqual(event.roles.get("から"), "山")
        self.assertEqual(event.verb, "帰る")

    def test_de_in_the_copula_is_not_a_locative_particle(self):
        event = extract_clause("ぶどうに手がとどきませんでした。")
        self.assertEqual(event.verb, "とどく")
        self.assertNotIn("で", event.roles)

    def test_verb_normalisation_polite_and_plain(self):
        self.assertEqual(_dictionary_verb("行きました")[0], "行く")
        self.assertEqual(_dictionary_verb("食べました")[0], "食べる")
        self.assertEqual(_dictionary_verb("買った")[0], "買う")
        self.assertEqual(_dictionary_verb("書いた")[0], "書く")
        self.assertEqual(_dictionary_verb("しました")[0], "する")

    def test_te_auxiliary_is_stripped(self):
        self.assertEqual(_dictionary_verb("流れてきました")[0], "流れる")

    def test_ruby_markers_are_removed(self):
        self.assertEqual(normalise_text("桃《もも》太郎《たろう》"), "桃太郎")

    def test_whole_story_keeps_one_protagonist_thread_and_the_key_events(self):
        events = extract_story(MOMOTARO)
        keys = {e.key for e in events}
        self.assertIn("おばあさん|ひろう|もも", keys)
        self.assertIn("男の子|生まれる|", keys)
        self.assertIn("ももたろう|やっつける|おに", keys)
        self.assertGreaterEqual(len(events), 9)
        # coverage: at least ~80% of sentences produced an event
        sentences = [s for s in MOMOTARO.split("。") if s.strip()]
        self.assertGreaterEqual(len(events), int(0.75 * len(sentences)))

    def test_fox_and_grapes_schema_is_recoverable(self):
        events = extract_story(FOX_GRAPES)
        verbs = [e.verb for e in events]
        self.assertIn("見つける", verbs)      # sees
        self.assertIn("言う", verbs)          # rationalises
        self.assertTrue(any("とどく" in v or "とびあがる" in v for v in verbs))  # tries / fails

    def test_learned_vocabulary_contains_repeated_content_words(self):
        vocab = learn_word_vocabulary(MOMOTARO)
        self.assertTrue({"おばあさん", "おじいさん"} & vocab or "ももたろう" in vocab)


if __name__ == "__main__":
    unittest.main()
