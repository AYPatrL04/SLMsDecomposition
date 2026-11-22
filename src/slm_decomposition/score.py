"""Score sealed generations; inference never loads the reference key."""
import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import load_config, read_json, sha256, write_json
from .prompts import answer, child_value, plan_slots
from .routing import combined, original_trace
from .source_check import compose_values, inspect_work, solve_source


def score_children(case, row, key, native):
    plan = row["plan"]["text"]
    truth = {c["name"].casefold(): c for c in key["children"]}
    certified_values = {}
    metrics = Counter()
    readouts = {}
    for layout, stages in (("flat", ("flat_left", "flat_right")),
                           ("nested", ("left_last", "right_last"))):
        correct_owners = set()
        generated = {}
        for stage in stages:
            parsed = child_value(row[stage]["text"])
            if parsed is None:
                continue
            who, value, unit = parsed
            expected = truth.get(who.casefold())
            if expected and unit.casefold() == expected["unit"].casefold():
                generated[who] = value
                if value == expected["value"]:
                    correct_owners.add(who.casefold())
        metrics[layout + "_correct_endpoints"] = len(correct_owners)
        metrics[layout + "_both_correct"] = int(len(correct_owners) == len(truth))
        readouts[layout] = compose_values(case, generated)

    for i, ((owner, unit), side) in enumerate(zip(plan_slots(plan), ("left", "right")), 1):
        if not owner or not unit:
            continue
        expected = truth.get(owner.casefold())
        carry = child_value(row[side + "_first"]["text"])
        if carry and expected:
            holder, value, carry_unit = carry
            boundary = expected["boundary"]
            typed = (holder.casefold(), carry_unit.casefold()) == (boundary["owner"].casefold(),
                                                                  expected["unit"].casefold())
            metrics["typed_carries"] += int(typed)
            metrics["correct_carries"] += int(typed and value == boundary["value"])
        trace = original_trace(case, combined(case, row, side), f"child{i}", plan)
        supported = inspect_work(case, trace, owner, unit)["supported_value"]
        if supported is not None:
            certified_values[owner] = supported

    comparison = compose_values(case, certified_values)
    draft_review = "\n".join(original_trace(case, row[s]["text"], s, plan)
                             for s in ("direct", "review"))
    certified_review = inspect_work(case, draft_review)["supported_value"]
    supported = {"cert_full": comparison, "cert_compact": comparison, "cert_review": certified_review}
    external = {k: native[k.removeprefix("cert_")] if v is None else v for k, v in supported.items()}
    external["symbolic"] = solve_source(case)
    return metrics, readouts, external, supported


def contrast(first, second, draws):
    differences = first - second
    wins = int(np.count_nonzero(differences == 1))
    losses = int(np.count_nonzero(differences == -1))
    discordant = wins + losses
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    probability = min(1.0, 2 * tail / (2 ** discordant))
    interval = np.quantile(differences[draws].mean(axis=1) * 100, [0.025, 0.975])
    return {"gains": wins, "losses": losses, "delta_pp": float(100 * differences.mean()),
            "ci95_pp": interval.tolist(), "exact_p": probability}


def paired_statistics(flags, directions, cfg):
    n = len(directions)
    seed, count = cfg["bootstrap_seed"], cfg["bootstrap_draws"]
    comparisons = [("full", "review"), ("full", "compact"), ("full", "flat_full"),
                   ("full", "direct"), ("direct", "frozen_direct"), ("review", "direct"),
                   ("cert_full", "cert_review"), ("cert_full", "cert_compact"), ("cert_full", "full")]
    if "compact_verified" in flags:
        comparisons.append(("compact_verified", "compact"))
    draws = np.random.default_rng(seed).integers(0, n, size=(count, n))
    pairs = {f"{a}_vs_{b}": contrast(flags[a], flags[b], draws) for a, b in comparisons}
    del draws

    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(directions == group) for group in sorted(set(directions))]
    draws = np.concatenate([rng.choice(s, size=(count, len(s))) for s in strata], axis=1)
    feedback = {}
    for a, b in (("full_no_check", "review_no_check"), ("full", "full_no_check"),
                 ("review", "review_no_check")):
        feedback[f"{a}_vs_{b}"] = contrast(flags[a], flags[b], draws)
    last = 0.0
    for i, result in enumerate(sorted(feedback.values(), key=lambda x: x["exact_p"])):
        last = max(last, min(1, (len(feedback) - i) * result["exact_p"]))
        result["holm_p"] = last
    effect = (flags["full"] - flags["full_no_check"]) - (flags["review"] - flags["review_no_check"])
    interaction = {"delta_pp": float(effect.mean() * 100),
                   "ci95_pp": np.quantile(effect[draws].mean(1) * 100, [0.025, 0.975]).tolist()}
    return pairs, feedback, interaction


def summarize(cases, rows, keys, config):
    if not cases or [c["id"] for c in cases] != [r["id"] for r in rows]:
        raise ValueError("Missing, duplicated or out-of-order assigned predictions")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate public input ID")
    arms = config["arms"]
    all_costs = dict(arms)
    all_costs.update(cert_full=arms["full"], cert_compact=arms["compact"],
                     cert_review=arms["review"], symbolic=[])
    stats = {name: Counter() for name in all_costs}
    flags = defaultdict(list)
    coverage = Counter()
    readout_merge = defaultdict(Counter)
    handoffs = defaultdict(Counter)
    directions = []
    batches = set()
    physical_seconds = 0.0

    for case, row in zip(cases, rows):
        key = keys[case["id"]]
        target = key["answer"]
        directions.append({"first_minus_second": "more", "second_minus_first": "fewer"}[key["direction"]])
        if set(row) != {"id", *config["caps"]}:
            raise ValueError(f"Missing or unexpected stages for {case['id']}")
        for stage, cap in config["caps"].items():
            obs = row[stage]
            if (obs["id"] != case["id"] or obs["stage"] != stage or obs["cap"] != cap
                    or not 0 <= obs["output_tokens"] <= cap):
                raise ValueError(f"Generation metadata mismatch for {case['id']} / {stage}")
            if obs["batch_id"] is not None and "reused_from" not in obs:
                batches.add(obs["batch_id"])
                physical_seconds += obs["amortized_seconds"]
        native = {arm: answer(row[arm]["text"]) for arm in arms}
        endpoints, readouts, external, supported = score_children(case, row, key, native)
        coverage.update(endpoints)
        for name, pred in (native | external).items():
            correct = pred == target
            flags[name].append(correct)
            bucket = stats[name]
            bucket["correct"] += correct
            bucket["valid"] += pred is not None
            if name in supported:
                bucket["source_supported"] += supported[name] is not None
            calls = [row[s] for s in all_costs[name]]
            bucket["calls"] += sum(c["batch_id"] is not None for c in calls)
            bucket["input_tokens"] += sum(c["input_tokens"] for c in calls)
            bucket["output_tokens"] += sum(c["output_tokens"] for c in calls)
            bucket["batch_seconds"] += sum(c["amortized_seconds"] for c in calls)
            bucket["missing_calls"] += sum(c["stop"] == "missing" for c in calls)
            bucket["cap_hits"] += sum(c["stop"] == "cap" for c in calls)
            bucket["terminal_cap_hits"] += bool(calls) and calls[-1]["stop"] == "cap"
        for layout, arm in (("flat", "flat_full"), ("nested", "full")):
            comparison_right = readouts[layout] == target
            merger_right = native[arm] == target
            cell = ("correct" if comparison_right else "wrong") + "_to_" + ("correct" if merger_right else "wrong")
            readout_merge[layout][cell] += 1
        stratum = "both_correct" if endpoints["nested_both_correct"] else "endpoint_error"
        outcome = {(True, True): "both", (True, False): "full_only",
                   (False, True): "compact_only", (False, False): "neither"}
        handoffs[stratum][outcome[(native["full"] == target, native["compact"] == target)]] += 1

    masks = {name: np.array(f, dtype=np.int8) for name, f in flags.items()}
    labels = np.array(directions)
    pairs, feedback, interaction = paired_statistics(masks, labels, config)
    def passes(comparison):
        return comparison["ci95_pp"][0] > 0 and comparison["exact_p"] < config["superiority_alpha"]
    full_panel = len(rows) == config["rows"]["panel"]
    decisions = None
    if full_panel:
        primary = passes(pairs["full_vs_review"])
        decisions = {"full_over_review": primary,
                     "full_over_compact": passes(pairs["full_vs_compact"]) if primary else None}
    return {"n": len(rows), "arms": stats, "paired": pairs, "feedback": feedback,
            "physical_batches": len(batches), "physical_batch_seconds": physical_seconds,
            "feedback_interaction": interaction, "endpoints": coverage, "readout_merge": readout_merge,
            "handoffs_by_endpoint_accuracy": handoffs, "ordered_superiority": decisions,
            "correct_by_direction": {d: {a: int(m[labels == d].sum()) for a, m in masks.items()}
                                     for d in sorted(set(directions))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    done = read_json(args.run / "completed.json")
    path = args.run / "outputs.json"
    if done["outputs_sha256"] != sha256(path):
        raise ValueError("Output digest does not match completion receipt")
    if "manifest" in done:
        manifest_path = args.run / "manifest.json"
        if sha256(manifest_path) != done["manifest"] or read_json(manifest_path)["config_sha256"] != sha256(args.config):
            raise ValueError("Scoring settings differ from the sealed inference run")
    inputs_path = cfg["data"]["inputs"]
    if "inputs_sha256" in done and done["inputs_sha256"] != sha256(inputs_path):
        raise ValueError("Configured public inputs differ from the generation inputs")
    cases = read_json(inputs_path)
    keys = read_json(cfg["data"]["references"])
    n = done["n"]
    if not 0 < n <= len(cases) or {row["id"] for row in cases} != keys.keys():
        raise ValueError("Incomplete public inputs or reference key")
    result = summarize(cases[:n], read_json(path), keys, cfg["evaluation"])
    write_json(args.run / "scores.json", result)
    for arm, metrics in result["arms"].items():
        print(f"{arm:16s}  {metrics['correct']:4d}/{n}   {metrics['output_tokens']:9d} output tokens"
              f"   {metrics['batch_seconds']:.2f} generation seconds")


if __name__ == "__main__":
    main()
