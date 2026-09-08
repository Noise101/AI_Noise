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

    def test_conjunction_keredo_is_not_read_as_a_subject(self):
        # 「けれども、…」 used to split into topic "けれど" + particle も and then
        # thread through the story as its subject.
        events = extract_story(
            "あひるさんは がっこうへ いきました。"
            "けれども、しかたがないので、その ようふくを きて いきました。"
            "けれども、がまんしました。")
        subs = [e.subject for e in events]
        self.assertNotIn("けれど", subs)
        self.assertNotIn("しかた", subs)
        self.assertIn("あひるさん", subs)

    def test_multi_clause_sentence_recovers_the_buried_subject_and_main_verb(self):
        # the が-marked subject sits mid-sentence behind a relative clause, the
        # main verb is three fragments later -- the old per-fragment parser got
        # neither
        events = extract_story(
            "あるとき、あそびまわっていたこうもりが、あやまって地べたにおちて、"
            "そこにいたいたちに、つかまってしまいました。")
        self.assertTrue(events)
        self.assertTrue(all(e.subject == "こうもり" for e in events))
        self.assertTrue(any(e.subject_explicit for e in events))
        self.assertIn("つかまる", [e.verb for e in events])

    def test_relative_clause_before_the_subject_is_stripped(self):
        event = extract_clause("そこにいたいたちがこうもりをつかまえました。")
        self.assertEqual(event.subject, "いたち")

    def test_a_name_is_never_split_by_the_modifier_stripper(self):
        from japanese_event_v1 import _strip_modifier
        self.assertEqual(_strip_modifier("ももたろう"), "ももたろう")

    def test_ga_at_the_end_of_a_comma_fragment_is_a_subject_marker(self):
        events = extract_story("おおきなたいこをもったたぬきが、やまからおりてきました。")
        self.assertEqual(events[-1].subject, "たぬき")

    def test_direct_speech_becomes_a_said_event(self):
        events = extract_story(
            "きつねは「たすけてください」とたのみました。"
            "くまが「よろしい」とこたえました。")
        speech = [e for e in events if e.roles.get("と")]
        self.assertEqual([(e.subject, e.verb) for e in speech],
                         [("きつね", "頼む"), ("くま", "答える")])
        self.assertIn("たすけて", speech[0].roles["と"])

    def test_speaker_can_follow_the_quote(self):
        events = extract_story("「おおい」と犬がさけびました。")
        self.assertEqual((events[0].subject, events[0].verb), ("犬", "叫ぶ"))

    def test_a_quotes_own_period_does_not_split_the_sentence(self):
        # the 。 inside 「…」 must not end the sentence early
        events = extract_story("むすめは「もう、いや。だいきらい。」といって、へやをでました。")
        verbs = [e.verb for e in events]
        self.assertIn("言う", verbs)
        self.assertTrue(any(v in ("でる", "出る") for v in verbs))

    def test_te_form_subject_carries_to_the_main_verb(self):
        events = extract_story("たくさんの牛があつまって、そうだん会をひらきました。")
        self.assertEqual(events[-1].subject, "牛")

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

    def test_verb_normalisation_negative_past_and_te_form(self):
        self.assertEqual(_dictionary_verb("できなかった")[0], "できる")
        self.assertEqual(_dictionary_verb("わからなかったので")[0], "わかる")
        self.assertEqual(_dictionary_verb("しなかった")[0], "する")
        self.assertEqual(_dictionary_verb("おちて")[0], "おちる")
        self.assertEqual(_dictionary_verb("つかまった")[0], "つかまる")
        self.assertEqual(_dictionary_verb("受けたのである")[0], "受ける")

    def test_verb_normalisation_compounds_and_copula(self):
        self.assertEqual(_dictionary_verb("いっておりました")[0], "言う")
        self.assertEqual(_dictionary_verb("見ながら言いました")[0], "言う")
        self.assertEqual(_dictionary_verb("かんがえました")[0], "考える")
        self.assertEqual(_dictionary_verb("くっつくものではなかった")[0], "くっつく")
        self.assertEqual(_dictionary_verb("泳ぐことができました")[0], "泳ぐ")
        self.assertEqual(_dictionary_verb("はできませんでした")[0], "できる")
        self.assertEqual(_dictionary_verb("あそんだ")[0], "あそぶ")
        # real verbs that start with a particle char are not truncated
        self.assertEqual(_dictionary_verb("はしる")[0], "はしる")
        self.assertEqual(_dictionary_verb("はいる")[0], "はいる")

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
