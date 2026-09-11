"""Explicit, content-addressed splits. Existing dataset preprocessing is unchanged."""
import json
import random
from adaptive_latent import content_hash, sample_id


def make_manifest(items, task, calibration, validation, holdout, seed=42,
                  historical_count=0, fingerprint=None):
    items = list(items)
    ids = [sample_id(item) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample content in source data")
    if min(calibration, validation, holdout, historical_count) < 0:
        raise ValueError("Split sizes cannot be negative")
    if calibration + validation + holdout + historical_count > len(items):
        raise ValueError("Not enough distinct source samples for the requested splits")
    historical = items[:historical_count]
    remaining = items[historical_count:]
    random.Random(seed).shuffle(remaining)
    splits = {"historical": historical, "calibration": remaining[:calibration],
              "validation": remaining[calibration:calibration + validation],
              "holdout": remaining[calibration + validation:calibration + validation + holdout]}
    doc = {"schema_version": 1, "task": task, "seed": seed,
           "source_fingerprint": fingerprint, "preprocessing": "existing_loader_prompt_metric_v2",
           "splits": {role: [{"sample_id": sample_id(item), "item": item} for item in rows]
                      for role, rows in splits.items()}}
    doc["manifest_hash"] = content_hash(doc)
    return doc


def load_manifest(path, task, split):
    with open(path, encoding="utf-8") as stream:
        doc = json.load(stream)
    payload = {k: v for k, v in doc.items() if k != "manifest_hash"}
    if doc.get("manifest_hash") != content_hash(payload):
        raise ValueError("Sample manifest hash mismatch")
    if doc["task"] != task:
        raise ValueError("Sample manifest task differs from requested task")
    seen = set()
    for rows in doc["splits"].values():
        for row in rows:
            identity = sample_id(row["item"])
            if identity != row["sample_id"] or identity in seen:
                raise ValueError("Sample ID mismatch or overlapping/duplicate splits")
            seen.add(identity)
    rows = doc["splits"][split]
    if not rows:
        raise ValueError(f"Empty split: {split}")
    return [row["item"] for row in rows], doc
