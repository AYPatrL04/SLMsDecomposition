"""Run the paired math policies on public questions and resume completed batches."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from filelock import FileLock
import torch

from .common import load_config, read_json, sha256, write_json
from .model import batch_run, encode_chat, load_adapter, load_model
from .routing import make_prompt


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_panel(path, limit=None):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array of public questions")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "document", "question"} or any(
            not isinstance(value, str) or not value.strip() for value in row.values()
        ):
            raise ValueError("Each question needs exactly three nonempty strings: id, document, question")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Question IDs must be unique")
    if limit is not None and not 0 < limit <= len(rows):
        raise ValueError("The item limit must be positive and within the panel size")
    return rows[:limit]


def register(directory, mode, config_path, inputs_path, adapter, ids):
    """Keep receipts tied to their inputs, executable method and study settings."""
    modules = ["common", "infer", "model", "lora", "prompts", "routing", "source_check"]
    if mode == "latency":
        modules.append("latency")
    code = {name: sha256(Path(__file__).with_name(name + ".py")) for name in modules}
    manifest = {
        "mode": mode, "config_sha256": sha256(config_path),
        "inputs_sha256": sha256(inputs_path), "adapter_sha256": sha256(adapter),
        "code": code, "ids": ids,
    }
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f"Incompatible run in {directory}. Choose a fresh output directory.")
    else:
        write_json(path, manifest)
    return sha256(path)


def record_runtime(directory, wrapper):
    """Keep partial measurements from different execution environments separate."""
    weight = next(wrapper.lm.parameters())
    device = weight.device
    runtime = {
        "device": device.type, "dtype": str(weight.dtype), "architecture": platform.machine(),
        "torch": str(torch.__version__), "transformers": version("transformers"),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_revision": getattr(wrapper.lm.config, "_commit_hash", None),
    }
    path = directory / "runtime.json"
    if path.exists() and read_json(path) != runtime:
        raise ValueError("Runtime changed for this partial run. Choose a fresh output directory.")
    write_json(path, runtime)


def missing(case_id, stage, cap):
    return {
        "id": case_id, "stage": stage, "text": "[missing generated item]", "stop": "missing",
        "eos": False, "cap": cap, "prompt_hash": digest(""), "input_tokens": 0,
        "output_tokens": 0, "batch_seconds": 0.0, "amortized_seconds": 0.0,
        "batch_size": 0, "batch_id": None,
    }


def cached_batch(requests, directory, wrapper, tokenizer, stage, cap, fingerprint):
    batch_key = digest(json.dumps([request["id"] for request in requests]))[:20]
    path = directory / f"{stage}-{batch_key}.json"
    split = path.with_suffix(".split")
    if path.exists():
        saved = read_json(path)
        rows = saved["records"]
        expected = [(r["id"], r["digest"], stage, cap) for r in requests]
        actual = [(r["id"], r["prompt_hash"], r["stage"], r["cap"]) for r in rows]
        if saved["manifest"] != fingerprint or actual != expected:
            raise ValueError(f"Stale batch receipt: {path}")
        return rows
    if not split.exists():
        try:
            rows = batch_run(requests, wrapper, tokenizer, stage, cap)
        except torch.OutOfMemoryError:
            if len(requests) == 1:
                raise
            split.touch()
        else:
            write_json(path, {"manifest": fingerprint, "records": rows})
            return rows

    # Keep the smaller layout after an OOM, even if the process is restarted.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mid = len(requests) // 2
    return [
        row for group in (requests[:mid], requests[mid:])
        for row in cached_batch(group, directory, wrapper, tokenizer, stage, cap, fingerprint)
    ]


def run(config_path, adapter, directory, inputs=None, limit=None):
    config = load_config(config_path)
    evaluation = config["evaluation"]
    panel_path = Path(inputs or config["data"]["inputs"])
    panel = public_panel(panel_path, limit)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".writer.lock"), timeout=0):
        fingerprint = register(directory, "inference", config_path, panel_path, adapter,
                               [row["id"] for row in panel])
        completed = directory / "completed.json"
        outputs = directory / "outputs.json"
        if completed.exists():
            receipt = read_json(completed)
            if receipt["manifest"] != fingerprint or receipt["outputs_sha256"] != sha256(outputs):
                raise ValueError("Completed outputs do not match the run receipt")
            print(f"Already complete: {outputs}", flush=True)
            return
        cache = directory / "batches"
        cache.mkdir(exist_ok=True)
        wrapper, tokenizer = load_model(config["model"])
        record_runtime(directory, wrapper)
        previous = {row["id"]: {} for row in panel}
        adapted = False
        batch_size = evaluation["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for stage, cap in evaluation["caps"].items():
            if stage != "frozen_direct" and not adapted:
                load_adapter(wrapper, adapter)
                adapted = True
            requests = []
            for case in panel:
                case_id = case["id"]
                prompt = make_prompt(stage, case, previous[case_id])
                if prompt is None:
                    previous[case_id][stage] = missing(case_id, stage, cap)
                    continue
                prompt_hash = digest(prompt)
                if stage.endswith("_verified"):
                    original = previous[case_id].get(stage.removesuffix("_verified"))
                    if original and (original["prompt_hash"], original["cap"]) == (prompt_hash, cap):
                        previous[case_id][stage] = {
                            **original, "stage": stage, "reused_from": original["stage"]
                        }
                        continue
                requests.append({"id": case_id, "digest": prompt_hash,
                                 "tokens": encode_chat(tokenizer, prompt)})
            requests.sort(key=lambda r: (len(r["tokens"]), r["id"]))
            for start in range(0, len(requests), batch_size):
                rows = cached_batch(requests[start:start + batch_size], cache, wrapper, tokenizer,
                                    stage, cap, fingerprint)
                for row in rows:
                    previous[row["id"]][stage] = row
            print(f"{stage}: {len(requests)}/{len(panel)} generated", flush=True)

        rows = [dict(id=case_id, **history) for case_id, history in previous.items()]
        write_json(outputs, rows)
        write_json(completed, {
            "manifest": fingerprint, "n": len(rows), "inputs_sha256": sha256(panel_path),
            "batches": len({r["batch_id"] for h in previous.values() for r in h.values()
                            if r["batch_id"] is not None}),
            "outputs_sha256": sha256(outputs),
        })
        print(f"Complete: {outputs}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Optional public JSON or .json.gz panel")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N assigned questions")
    args = parser.parse_args()
    run(args.config, args.adapter, args.output, args.inputs, args.limit)


if __name__ == "__main__":
    main()
