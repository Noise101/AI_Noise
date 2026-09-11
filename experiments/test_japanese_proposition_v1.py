import unittest

import japanese_proposition_v1 as jp


class JapanesePropositionTests(unittest.TestCase):
    def assert_prop(self, text, expected):
        prop = jp.extract_proposition(text)
        self.assertIsNotNone(prop, text)
        self.assertEqual((prop.subject, prop.relation, prop.value), expected)
        self.assertEqual(prop.provenance, "proposition_self")

    def test_nominal_definition(self):
        self.assert_prop("猫は動物です。", ("猫", "is", "動物"))
        self.assert_prop("レモンとは果物です。", ("レモン", "is", "果物"))
        self.assert_prop("絵画は装飾品ではない。", ("絵画", "is-not", "装飾品"))

    def test_adjective_state_and_negation(self):
        self.assert_prop("レモンは黄色い。", ("レモン", "is", "黄色い"))
        self.assert_prop("花子は悲しくなかった。", ("花子", "is-not", "悲しい"))

    def test_possession_and_location(self):
        self.assert_prop("犬には四本の足がある。", ("犬", "has", "足"))
        self.assert_prop("本は机の上にある。", ("本", "at", "上"))

    def test_action_clause_is_not_a_proposition(self):
        self.assertIsNone(jp.extract_proposition("太郎はりんごを食べた。"))

    def test_later_predicate_in_a_long_clause_is_not_attached_to_the_topic(self):
        self.assertIsNone(jp.extract_proposition(
            "私はすぐにも命を取られるから、静かになんぞしていられない。"))
        self.assertIsNone(jp.extract_proposition(
            "つるは敵を討ちたいと思い、二日たって狐を呼びました。"))
        self.assertIsNone(jp.extract_proposition("お前は何者だ？"))
        self.assertIsNone(jp.extract_proposition("お前はだれだ。"))
        self.assertIsNone(jp.extract_proposition(
            "王女は世界中で一番美しい人にそういありません。"))

    def test_story_extracts_multiple_stative_observations(self):
        props = jp.extract_story("猫は動物です。猫は小さい。猫は魚を食べた。")
        self.assertEqual([p.value for p in props], ["動物", "小さい"])


if __name__ == "__main__":
    unittest.main()
