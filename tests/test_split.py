"""Check the fixed public inputs, disjoint training allocation and scoring key together."""

import hashlib
import json
from pathlib import Path

from slm_decomposition.common import load_config, read_json
from slm_decomposition.source_check import solve_source
from slm_decomposition.train import training_examples


def test_published_allocation():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/study.yaml")
    data = config["data"]
    public = read_json(data["inputs"])
    keys = read_json(data["references"])
    train = read_json(data["train"])
    ids = {row["id"] for row in public}
    assert len(ids) == len(public) == config["evaluation"]["rows"]["panel"]
    assert ids == keys.keys() and ids.isdisjoint(row["id"] for row in train)
    assert all(set(row) == {"id", "question", "document"} for row in public)
    assert all(solve_source(row) == keys[row["id"]]["answer"] for row in public)

    examples = training_examples(data["train"], config["training"])
    mixture = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()
    assert len(examples) == 2688
    assert mixture == "05f28ccef3e26d378c02643d5419f8b2ea864a50f054e197abc28591427a6f9e"
