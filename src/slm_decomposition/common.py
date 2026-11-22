"""Configuration and local artifacts for the MathGAP experiment."""
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml


def load_config(path):
    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["data"] = {
        key: str((source.parent / Path(value).expanduser()).resolve())
        for key, value in cfg["data"].items()
    }
    evaluation = cfg.get("evaluation", {})
    verified = evaluation.get("verified_compact", False)
    if not isinstance(verified, bool):
        raise ValueError("verified_compact must be a boolean")
    if verified:
        original = ("left_last", "right_last", "compact")
        caps = evaluation["caps"]
        caps.update({f"{stage}_verified": caps[stage] for stage in original})
        arms = evaluation["arms"]
        arms["compact_verified"] = [
            f"{stage}_verified" if stage in original else stage for stage in arms["compact"]
        ]
    return cfg


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
