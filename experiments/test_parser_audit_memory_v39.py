import unittest

from parser_audit_memory_v39 import empty_audit_memory, ingest_report


class ParserAuditMemoryTest(unittest.TestCase):
    def test_keeps_rejection_without_admitting_it_as_knowledge(self):
        report = {"knowledge": {"bootstrap": {"sources": [{"url": "https://story",
            "event_extraction_audit": [
                {"sentence": "Fox ran.", "accepted": True, "event": "fox|ran|", "reason": "accepted"},
                {"sentence": "And through woods.", "accepted": False, "event": None,
                 "reason": "no_explicit_action"}]}]}}}
        memory = empty_audit_memory()
        ingest_report(memory, "fox", report)
        self.assertEqual(memory["summary"]["quarantined_rejections"], 1)
        self.assertEqual(memory["summary"]["admitted_as_world_knowledge"], 0)
        rejected = [item for item in memory["records"].values() if item["quarantined"]]
        self.assertEqual(rejected[0]["sentence"], "And through woods.")


if __name__ == "__main__":
    unittest.main()
