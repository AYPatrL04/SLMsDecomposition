from hashlib import sha256
import json
from pathlib import Path

import pytest

from slm_decomposition.common import load_config, read_json
from slm_decomposition.latency import select


def test_release_keeps_ordered_latency_sample():
    config = load_config(Path(__file__).parents[1] / "configs" / "study.yaml")
    panel = read_json(config["data"]["inputs"])
    assignments = select(panel, config["latency"]["assignments"])
    payload = json.dumps(assignments, separators=(",", ":")).encode()
    assert len(assignments) == 24
    assert sha256(payload).hexdigest() == "c8278bf1c22e24f7a0457ff919912dba94f6f82f51c9144c8f178bbe6a4a6420"


def test_latency_assignments_reject_bad_ids_and_flags():
    panel = [{"id": "sample"}]
    row = {"id": "sample", "whole_first": True, "full_first": False}
    with pytest.raises(ValueError, match="Unknown or repeated"):
        select(panel, [row, row])
    with pytest.raises(ValueError, match="Unknown or repeated"):
        select(panel, [{**row, "id": "missing"}])
    with pytest.raises(ValueError, match="flags must be booleans"):
        select(panel, [{**row, "whole_first": 1}])
