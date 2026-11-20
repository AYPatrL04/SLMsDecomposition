"""Replay the registered, counterbalanced sample with one warm model request at a time."""

import argparse
from pathlib import Path
import statistics
import time

from filelock import FileLock

from .common import load_config, read_json, sha256, write_json
from .infer import digest, missing, public_panel, record_runtime, register
from .model import batch_run, encode_chat, load_adapter, load_model, synchronize
from .prompts import answer
from .routing import make_prompt


def select(panel, registered):
    """Read the published question IDs and execution order from the configuration."""
    if not isinstance(registered, list) or not registered:
        raise ValueError("Latency assignments must be a nonempty list")
    fields = {"id", "whole_first", "full_first"}
    known = {case["id"] for case in panel}
    seen = set()
    assignments = []
    for row in registered:
        if not isinstance(row, dict) or row.keys() != fields:
            raise ValueError("A latency assignment needs an ID and both execution-order flags")
        case_id = row["id"]
        if not isinstance(case_id, str) or case_id not in known or case_id in seen:
            raise ValueError(f"Unknown or repeated latency question: {case_id!r}")
        whole_first, full_first = row["whole_first"], row["full_first"]
        if type(whole_first) is not bool or type(full_first) is not bool:
            raise ValueError(f"Latency order flags must be booleans for {case_id}")
        assignments.append((case_id, whole_first, full_first))
        seen.add(case_id)
    return assignments


def schedule(arms, whole_first, full_first):
    whole = arms["review"]
    first, second = ("full", "compact") if full_first else ("compact", "full")
    split = list(dict.fromkeys([*arms[first], *arms[second]]))
    return whole + split if whole_first else split + whole


def warmup():
    names = "Ada Beth Chad Dale Evan Finn Glen Hugh Iris Judy Kane Lara Mina Noah Opal Paul Quin Rita".split()
    statements = []
    for chain in (names[:9], names[9:]):
        statements.append(f"{chain[0]} has 10 rugs.")
        for i, (prev, name) in enumerate(zip(chain, chain[1:])):
            direction = "more" if i % 2 == 0 else "fewer"
            statements.append(f"{name} has 2 rugs {direction} than {prev}.")
    return {"id": "warmup-scope-prior", "document": "\n".join(reversed(statements)),
            "question": "How many rugs does Iris have more than Rita?"}


def time_stage(case, history, stage, cap, wrapper, tokenizer):
    device = next(wrapper.lm.parameters()).device
    synchronize(device)
    started = time.perf_counter()
    prompt = make_prompt(stage, case, history)
    if prompt is None:
        row = missing(case["id"], stage, cap)
    else:
        request = {"id": case["id"], "digest": digest(prompt), "tokens": encode_chat(tokenizer, prompt)}
        row = batch_run([request], wrapper, tokenizer, stage, cap)[0]
    synchronize(device)
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def summarize(rows, arms, primary, references):
    summary = {}
    timings = {}
    for arm in ("direct", "review", "full", "compact"):
        stages = arms[arm]
        timings[arm] = [sum(row[s]["elapsed_seconds"] for s in stages) for row in rows]
        times = timings[arm]
        # Same linear interpolation as the original empirical 90th percentile.
        p90 = statistics.quantiles(times, n=10, method="inclusive")[8] if len(times) > 1 else times[0]
        stats = {"mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
                 "p90_seconds": p90, "parents": len(rows)}
        for key in ("input_tokens", "output_tokens"):
            stats[key] = sum(row[s][key] for row in rows for s in stages)
        correct = 0
        agreement = 0
        for row in rows:
            key = row["id"]
            prediction = answer(row[stages[-1]]["text"])
            correct += prediction == references[key]["answer"]
            if primary and prediction is not None:
                agreement += prediction == answer(primary[key][stages[-1]]["text"])
        stats["correct"] = correct
        if primary:
            stats["same_valid_as_primary"] = agreement
        summary[arm] = stats
    pairs = {}
    for other in ("direct", "review", "compact"):
        differences = [a - b for a, b in zip(timings["full"], timings[other])]
        pairs[other] = {"mean_full_minus_other_seconds": statistics.mean(differences),
                        "full_faster": sum(delta < 0 for delta in differences)}
    return {"policies": summary, "paired_to_full": pairs}


def run(config_path, adapter, output, primary_dir=None, limit=None):
    config = load_config(config_path)
    evaluation = config["evaluation"]
    spec = config["latency"]
    caps, arms = evaluation["caps"], evaluation["arms"]
    panel_path = Path(config["data"]["inputs"])
    panel = public_panel(panel_path)
    by_id = {row["id"]: row for row in panel}
    assignments = select(panel, spec["assignments"])
    if limit is not None and not 0 < limit <= len(assignments):
        raise ValueError("Latency limit must be within the registered sample")
    assignments = assignments[:limit]
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".writer.lock"), timeout=0):
        fingerprint = register(output, "latency", config_path, panel_path, adapter, [a[0] for a in assignments])
        outputs = output / "outputs.json"
        completed = output / "completed.json"
        if completed.exists():
            record = read_json(completed)
            if record["manifest"] != fingerprint or record["outputs_sha256"] != sha256(outputs):
                raise ValueError("Latency output does not match completion receipt")
        else:
            wrapper, tokenizer = load_model(config["model"])
            record_runtime(output, wrapper)
            load_adapter(wrapper, adapter)
            case = warmup()
            history = {}
            for stage in schedule(arms, True, True):
                history[stage] = time_stage(case, history, stage, caps[stage], wrapper, tokenizer)
            rows = []
            for case_id, whole_first, full_first in assignments:
                case, history = by_id[case_id], {}
                for stage in schedule(arms, whole_first, full_first):
                    path = output / "stages" / digest(case_id) / f"{stage}.json"
                    if path.exists():
                        receipt = read_json(path)
                        row = receipt["record"]
                        if (receipt["manifest"] != fingerprint or row["id"] != case_id or
                            row["stage"] != stage or row["prompt_hash"] != digest(make_prompt(stage, case, history) or "")):
                            raise ValueError(f"Incompatible latency receipt: {path}")
                    else:
                        row = time_stage(case, history, stage, caps[stage], wrapper, tokenizer)
                        write_json(path, {"manifest": fingerprint, "record": row})
                    history[stage] = row
                rows.append(dict(id=case_id, **history))
                print(f"Timed {len(rows)}/{len(assignments)}", flush=True)
            write_json(outputs, rows)
            write_json(completed, {"manifest": fingerprint, "n": len(rows),
                                   "outputs_sha256": sha256(outputs), "inputs_sha256": sha256(panel_path)})

        # Open keys or primary predictions only after the latency generations are sealed.
        primary = None
        if primary_dir:
            original = read_json(primary_dir / "completed.json")
            original_manifest = read_json(primary_dir / "manifest.json")
            current = read_json(output / "manifest.json")
            if (original["outputs_sha256"] != sha256(primary_dir / "outputs.json") or
                original["manifest"] != sha256(primary_dir / "manifest.json") or
                original_manifest["mode"] != "inference" or
                any(current["code"].get(k) != v for k, v in original_manifest["code"].items()) or
                any(original_manifest[k] != current[k] for k in ("inputs_sha256", "adapter_sha256", "config_sha256"))):
                raise ValueError("The primary run is incompatible with this latency replay")
            primary = {row["id"]: row for row in read_json(primary_dir / "outputs.json")}
            if any(case_id not in primary for case_id, _, _ in assignments):
                raise ValueError("The primary run does not cover the latency sample")
        results = summarize(read_json(outputs), arms, primary, read_json(config["data"]["references"]))
        write_json(output / "summary.json", results)
        print(f"Latency summary: {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", type=Path, help="Optional primary inference directory for batch-size agreement")
    parser.add_argument("--limit", type=int, help="Time only the first N assigned sample parents")
    args = parser.parse_args()
    run(args.config, args.adapter, args.output, args.primary, args.limit)


if __name__ == "__main__":
    main()
