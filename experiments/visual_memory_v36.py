#!/usr/bin/env python3
"""Source-aware image observation and association without a pretrained vision model."""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import urllib.parse
from pathlib import Path

from PIL import Image

from web_cache import WEB_CACHE


USER_AGENT = "AI_Noise/0.36 (read-only visual development experiment)"
ACQUISITION_INTERVAL_SECONDS = 60


def empty_visual_memory() -> dict:
    return {"version": 36, "mode": "depiction_observation", "pending_seeds": [],
            "completed_seeds": [], "observations": {}, "associations": [],
            "near_duplicate_groups": [], "last_acquired_epoch": 0.0, "summary": {},
            "warning": "a web image is an observed depiction, not physical contact or proof of its label"}


def enqueue(memory: dict, seeds: list[str]) -> None:
    known = set(memory.get("pending_seeds", [])) | set(memory.get("completed_seeds", []))
    pending = memory.setdefault("pending_seeds", [])
    for seed in seeds:
        if seed and seed not in known:
            pending.append(seed)
            known.add(seed)


def _external(metadata: dict, name: str) -> str | None:
    value = metadata.get(name, {})
    return value.get("value") if isinstance(value, dict) else None


class CommonsProvider:
    API = "https://commons.wikimedia.org/w/api.php"

    def search(self, query: str, limit: int = 5) -> list[dict]:
        params = urllib.parse.urlencode({"action": "query", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": limit,
            "prop": "imageinfo", "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 256,
            "format": "json", "formatversion": 2})
        data = WEB_CACHE.get_json(self.API + "?" + params, USER_AGENT)
        results = []
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata", {})
            license_name = _external(metadata, "LicenseShortName")
            thumb = info.get("thumburl")
            if not thumb or not license_name or not info.get("mime", "").startswith("image/"):
                continue
            results.append({"title": page.get("title"), "page_url": info.get("descriptionurl"),
                "thumbnail_url": thumb, "original_url": info.get("url"),
                "mime": info.get("mime"), "width": info.get("width"), "height": info.get("height"),
                "license": license_name, "license_url": _external(metadata, "LicenseUrl"),
                "creator": _external(metadata, "Artist"),
                "description": _external(metadata, "ImageDescription"),
                "categories": _external(metadata, "Categories")})
        return results

    def fetch(self, url: str) -> bytes:
        return WEB_CACHE.get_bytes(url, USER_AGENT, "image/*")


def image_features(payload: bytes) -> tuple[dict, bytes]:
    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
        image.thumbnail((256, 256))
        width, height = image.size
        sample = image.resize((32, 32))
        pixels = list(sample.getdata())
        means = [sum(pixel[channel] for pixel in pixels) / len(pixels) for channel in range(3)]
        histogram = []
        for channel in range(3):
            counts = [0, 0, 0, 0]
            for pixel in pixels:
                counts[min(3, pixel[channel] // 64)] += 1
            histogram.extend(round(count / len(pixels), 4) for count in counts)
        gray = sample.convert("L")
        values = list(gray.getdata())
        edge_pairs = edge_hits = 0
        for y in range(32):
            for x in range(32):
                here = values[y * 32 + x]
                for dx, dy in ((1, 0), (0, 1)):
                    if x + dx < 32 and y + dy < 32:
                        edge_pairs += 1
                        edge_hits += abs(here - values[(y + dy) * 32 + x + dx]) >= 24
        tiny = gray.resize((8, 8))
        tiny_values = list(tiny.getdata())
        average = sum(tiny_values) / len(tiny_values)
        perceptual_hash = f"{sum((value >= average) << index for index, value in enumerate(tiny_values)):016x}"
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80, optimize=True)
        return ({"width": width, "height": height,
                 "aspect_ratio": round(width / max(1, height), 4),
                 "mean_rgb": [round(value, 2) for value in means],
                 "rgb_histogram_4_bins": histogram,
                 "edge_density": round(edge_hits / max(1, edge_pairs), 4),
                 "perceptual_hash": perceptual_hash}, output.getvalue())


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def rebuild_associations(memory: dict) -> None:
    observations = list(memory.get("observations", {}).values())
    associations = []
    for item in observations:
        associations.append({"source": f"word_or_phrase:{item['query']}",
                             "target": f"depiction:{item['observation_id']}",
                             "kind": "external_search_context",
                             "status": "unverified_metadata_association",
                             "causal_credit": False})
    groups = []
    ungrouped = set(item["observation_id"] for item in observations)
    by_id = {item["observation_id"]: item for item in observations}
    while ungrouped:
        first = min(ungrouped)
        ungrouped.remove(first)
        group = [first]
        first_hash = by_id[first]["features"]["perceptual_hash"]
        for candidate in sorted(ungrouped):
            if hamming(first_hash, by_id[candidate]["features"]["perceptual_hash"]) <= 5:
                group.append(candidate)
        ungrouped.difference_update(group[1:])
        if len(group) > 1:
            groups.append({"members": group, "status": "near_duplicate_not_independent_experience"})
    memory["associations"] = associations
    memory["near_duplicate_groups"] = groups
    memory["summary"] = summarize(memory)


def summarize(memory: dict) -> dict:
    observations = list(memory.get("observations", {}).values())
    return {"mode": memory.get("mode", "depiction_observation"),
            "pending_visual_curricula": len(memory.get("pending_seeds", [])),
            "completed_visual_curricula": len(memory.get("completed_seeds", [])),
            "depictions_seen": len(observations),
            "physical_objects_seen": 0,
            "near_duplicate_groups": len(memory.get("near_duplicate_groups", [])),
            "grounded_visual_concepts": 0,
            "decision_influence": False}


def acquire_one(memory: dict, image_dir: Path, provider: CommonsProvider | None = None,
                current_epoch: float | None = None, force: bool = False) -> dict:
    current_epoch = time.time() if current_epoch is None else current_epoch
    if (not force and current_epoch - memory.get("last_acquired_epoch", 0)
            < ACQUISITION_INTERVAL_SECONDS):
        memory["summary"] = summarize(memory)
        return {"status": "cooldown", **memory["summary"]}
    pending = memory.get("pending_seeds", [])
    if not pending:
        memory["summary"] = summarize(memory)
        return {"status": "no_visual_gap", **memory["summary"]}
    seed = pending.pop(0)
    provider = provider or CommonsProvider()
    WEB_CACHE.set_network_budget(3)
    try:
        results = provider.search(seed, 5)
    except Exception as error:
        pending.insert(0, seed)
        memory["last_acquired_epoch"] = current_epoch
        memory["summary"] = summarize(memory)
        return {"status": "visual_source_retry_wait", "seed": seed,
                "error": f"{type(error).__name__}: {error}", **memory["summary"]}
    accepted = None
    for candidate in results:
        try:
            payload = provider.fetch(candidate["thumbnail_url"])
            features, normalized = image_features(payload)
        except Exception:
            continue
        digest = hashlib.sha256(normalized).hexdigest()
        observation_id = digest[:20]
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{digest}.jpg"
        if not path.exists():
            path.write_bytes(normalized)
        memory.setdefault("observations", {})[observation_id] = {
            "observation_id": observation_id, "query": seed,
            "experience_kind": "web_depiction_seen", "physical_object_seen": False,
            "grounding_status": "unverified_metadata_association",
            "source": candidate, "local_thumbnail": str(path), "features": features,
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current_epoch)),
        }
        accepted = observation_id
        break
    memory.setdefault("completed_seeds", []).append(seed)
    memory["last_acquired_epoch"] = current_epoch
    rebuild_associations(memory)
    return {"status": "depiction_observed" if accepted else "no_usable_depiction",
            "seed": seed, "observation_id": accepted, **memory["summary"]}
