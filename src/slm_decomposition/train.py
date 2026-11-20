"""Train the single shared adapter from the locked short-chain parent problems."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from filelock import FileLock
import torch
from torch.nn import functional as F

from . import lora
from .common import load_config, read_json, sha256, write_json
from .model import encode_chat, load_model, synchronize
from .training_data import assemble


def training_examples(source, spec):
    parents = read_json(source)
    if len(parents) != spec["parents"] or len({row["id"] for row in parents}) != len(parents):
        raise ValueError("Training parent count or IDs differ from the configured allocation")
    parents.sort(key=lambda row: hashlib.sha256(("mix|" + row["id"]).encode()).digest())
    return [example for i, row in enumerate(parents)
            for example in assemble(row, i, spec["planner_every_parents"])]


def tokenize(tokenizer, examples, limit):
    encodings = []
    for row in examples:
        prefix = encode_chat(tokenizer, row["prompt"])
        target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        if not prefix or len(prefix) + len(target) > limit:
            raise ValueError(f"Training example outside the configured token limit: {row['id']}")
        encodings.append((prefix, target))
    return encodings


def target_loss(model, example):
    """Mean cross entropy over one response, including EOS; prompts receive no loss."""
    prefix, target = example
    device = next(model.parameters()).device
    ids = torch.tensor([prefix + target[:-1]], device=device)
    states = model.lm.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False, return_dict=True).last_hidden_state
    logits = model.lm.lm_head(states[:, -len(target):]).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(target, device=device))


def train(config, source, output, max_updates=None):
    if max_updates is not None and max_updates < 1:
        raise ValueError("--max-updates must be positive")
    spec = config["training"]
    seed = spec["seed"]
    examples = training_examples(source, spec)
    mixture = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()
    counts = Counter(row["id"].rsplit("/", 1)[-1] for row in examples)
    print(f"Prepared {len(examples)} training examples; SHA-256 {mixture}", flush=True)
    torch.manual_seed(seed)
    model, tok = load_model(config["model"])
    torch.manual_seed(seed)
    layout = lora.attach(model, spec)
    encodings = tokenize(tok, examples, spec["max_sequence_tokens"])
    if spec["gradient_checkpointing"]:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.lm.enable_input_require_grads()
    model.lm.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    accumulation = spec["accumulation"]
    total = math.ceil(len(examples) / accumulation) * spec["epochs"]
    stop = total if max_updates is None else min(total, max_updates)
    device = next(model.parameters()).device
    initial = lora.adapter_state(model)
    started = time.perf_counter()
    history = []
    for epoch in range(spec["epochs"]):
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        for index in range(0, len(order), accumulation):
            group = order[index:index + accumulation]
            step = len(history) + 1
            warmup = spec["warmup_updates"]
            if step <= warmup:
                fraction = step / warmup
            else:
                fraction = 1 - (1 - spec["final_lr_fraction"]) * (step - warmup) / (total - warmup)
            rate = spec["learning_rate"] * fraction
            optimizer.param_groups[0]["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for j in group:
                loss = target_loss(model, encodings[j]) / len(group)
                loss.backward()
                group_loss += float(loss.detach())
            if not math.isfinite(group_loss):
                raise FloatingPointError(f"Nonfinite target loss at update {step}")
            norm = torch.nn.utils.clip_grad_norm_(trainable, spec["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            history.append({"update": step, "loss": group_loss, "gradient_norm": float(norm),
                            "learning_rate": rate})
            if step == 1 or step % 24 == 0 or step == stop:
                print(f"Update {step}/{stop}: target loss {group_loss:.4f}", flush=True)
            if step == stop:
                break
        if len(history) == stop:
            break
    synchronize(device)
    elapsed = time.perf_counter() - started
    state = lora.adapter_state(model)
    change = math.sqrt(sum(float((p - initial[k]).double().square().sum()) for k, p in state.items()))
    if not math.isfinite(change) or change <= 0:
        raise RuntimeError("Adapter parameters did not acquire a finite update")
    model.lm.eval()
    checkpoint = {"adapter": state, "metadata": layout, "training_spec": spec,
                  "model_id": config["model"]["checkpoint"], "revision": config["model"]["revision"],
                  "training_examples_sha256": mixture, "updates_completed": len(history)}
    name = "adapter" if stop == total else "smoke-adapter"
    temporary = output / (name + ".tmp")
    torch.save(checkpoint, temporary)
    lora.restore(model, torch.load(temporary, weights_only=True, map_location="cpu")["adapter"])
    receipt = {
        "model": checkpoint["model_id"], "revision": checkpoint["revision"], "seed": seed,
        "parents": spec["parents"], "examples": len(examples), "tasks": counts,
        "training_examples_sha256": mixture, "updates": len(history), "planned_updates": total,
        "trainable_parameters": layout["trainable_parameters"], "adapter_l2_change": change,
        "training_input_tokens": sum(len(prefix) for prefix, _ in encodings),
        "training_target_tokens": sum(len(target) for _, target in encodings),
        "device": str(device), "seconds": elapsed, "history": history,
        "checkpoint_sha256": sha256(temporary),
    }
    write_json(output / (name + "-receipt.json"), receipt)
    os.replace(temporary, output / (name + ".pt"))
    print(f"Saved {output / (name + '.pt')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--data", type=Path, help="Override the configured training set (JSON or JSON.gz)")
    parser.add_argument("--output", type=Path, default=Path("runs/train"))
    parser.add_argument("--max-updates", type=int, help="Stop early for a local smoke test")
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".train.lock"), timeout=0):
        if any(args.output.glob("*.pt")):
            raise FileExistsError(f"An adapter already exists in {args.output}. Use a new output directory.")
        train(config, args.data or Path(config["data"]["train"]), args.output, args.max_updates)


if __name__ == "__main__":
    main()
