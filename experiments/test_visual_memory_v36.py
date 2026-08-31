import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from visual_memory_v36 import (acquire_one, empty_visual_memory, enqueue, hamming,
                               image_features, rebuild_associations)


def image_bytes(color=(220, 210, 20)):
    output = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(output, format="PNG")
    return output.getvalue()


class FakeProvider:
    def search(self, query, limit=5):
        return [{"title": "File:Example.jpg", "page_url": "https://commons/page",
                 "thumbnail_url": "https://commons/thumb", "original_url": "https://commons/original",
                 "mime": "image/jpeg", "width": 400, "height": 300,
                 "license": "CC BY-SA 4.0", "license_url": "https://license",
                 "creator": "Example", "description": query, "categories": "Example"}]

    def fetch(self, url):
        return image_bytes()


class VisualMemoryTest(unittest.TestCase):
    def test_extracts_local_features_without_semantic_label(self):
        features, normalized = image_features(image_bytes())
        self.assertEqual(features["width"], 40)
        self.assertEqual(len(features["rgb_histogram_4_bins"]), 12)
        self.assertTrue(normalized.startswith(b"\xff\xd8"))

    def test_observation_distinguishes_depiction_from_physical_object(self):
        memory = empty_visual_memory()
        enqueue(memory, ["lemon"])
        with tempfile.TemporaryDirectory() as directory:
            result = acquire_one(memory, Path(directory), FakeProvider(), 1000, force=True)
        observation = next(iter(memory["observations"].values()))
        self.assertEqual(result["status"], "depiction_observed")
        self.assertFalse(observation["physical_object_seen"])
        self.assertEqual(observation["grounding_status"], "unverified_metadata_association")
        self.assertEqual(memory["summary"]["decision_influence"], False)

    def test_queue_deduplicates_language_curricula(self):
        memory = empty_visual_memory()
        enqueue(memory, ["lemon", "lemon"])
        enqueue(memory, ["lemon"])
        self.assertEqual(memory["pending_seeds"], ["lemon"])

    def test_near_duplicate_images_are_not_independent_experience(self):
        features, _ = image_features(image_bytes())
        memory = empty_visual_memory()
        memory["observations"] = {
            "one": {"observation_id": "one", "query": "lemon", "features": features},
            "two": {"observation_id": "two", "query": "fruit", "features": features},
        }
        rebuild_associations(memory)
        self.assertEqual(len(memory["near_duplicate_groups"]), 1)
        self.assertEqual(hamming(features["perceptual_hash"], features["perceptual_hash"]), 0)


if __name__ == "__main__":
    unittest.main()
