"""Load one text decoder and run the study's thinking-off greedy generation."""

from contextlib import nullcontext
import hashlib
import json
import time

import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForCausalLM

from . import lora


class TextModel(torch.nn.Module):
    """Keep the lm.* parameter names used by the original arithmetic adapter."""

    def __init__(self, lm, model_id, revision):
        super().__init__()
        self.lm = lm.requires_grad_(False).eval()
        self.model_id = model_id
        self.revision = revision


def load_model(options):
    requested = options["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    dtype = options["dtype"]
    if dtype == "auto":
        dtype = "float32"
        if device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_bf16_supported():
                    dtype = "bfloat16"
    hub = {"revision": options["revision"], "local_files_only": options["local_files_only"]}
    checkpoint = options["checkpoint"]
    tok = AutoTokenizer.from_pretrained(checkpoint, **hub)
    if tok.eos_token_id is None:
        raise ValueError("The tokenizer has no EOS token")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # Generation below constructs left-padding explicitly.
    lm = Qwen3_5ForCausalLM.from_pretrained(checkpoint, dtype=getattr(torch, dtype),
                                             attn_implementation=options["attn_implementation"], **hub)
    return TextModel(lm.to(device), checkpoint, options["revision"]), tok


def encode_chat(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )


def load_adapter(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for field in ("model_id", "revision"):
        if field in checkpoint and checkpoint[field] != getattr(model, field):
            raise ValueError(f"Adapter {field} does not match the configured base model")
    layout = lora.attach(model, checkpoint["training_spec"])
    if layout != checkpoint["metadata"]:
        raise ValueError("The saved adapter layout does not match the decoder")
    lora.restore(model, checkpoint["adapter"])
    model.eval()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def batch_run(requests, wrapper, tokenizer, stage, cap):
    """Charge the synchronized generation once per batch, including EOS in token counts."""
    lm = wrapper.lm
    device = next(lm.parameters()).device
    width = max(len(r["tokens"]) for r in requests)
    if width + cap > lm.config.max_position_embeddings:
        raise ValueError("Prompt and generation allowance exceed the model context")
    pad = tokenizer.pad_token_id
    padded = [[pad] * (width - len(r["tokens"])) + r["tokens"] for r in requests]
    masks = [[0] * (width - len(r["tokens"])) + [1] * len(r["tokens"]) for r in requests]
    ids = torch.tensor(padded, device=device)
    mask = torch.tensor(masks, device=device)
    configured = lm.generation_config.eos_token_id
    ends = list(configured) if isinstance(configured, (list, tuple)) else [configured]
    ends = list(dict.fromkeys(x for x in [*ends, tokenizer.eos_token_id] if x is not None))
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=cap,
                                  eos_token_id=ends, pad_token_id=pad, use_cache=True)
    with lora.disabled(wrapper) if stage == "frozen_direct" else nullcontext():
        synchronize(device)
        started = time.perf_counter()
        output = lm.generate(input_ids=ids, attention_mask=mask, generation_config=generation,
                             logits_to_keep=1)
        synchronize(device)
        seconds = time.perf_counter() - started

    keys = [hashlib.sha256(r["id"].encode()).hexdigest()[:20] + "-" + stage for r in requests]
    batch_id = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    records = []
    for request, sequence in zip(requests, output[:, width:].tolist()):
        stop = next((index + 1 for index, token in enumerate(sequence) if token in ends), None)
        used = sequence[:stop]
        records.append({
            "id": request["id"], "stage": stage, "text": tokenizer.decode(used, skip_special_tokens=True),
            "eos": stop is not None, "stop": "eos" if stop is not None else "cap", "cap": cap,
            "prompt_hash": request["digest"], "input_tokens": len(request["tokens"]),
            "output_tokens": len(used), "batch_id": batch_id, "batch_size": len(requests),
            "batch_seconds": seconds, "amortized_seconds": seconds / len(requests),
        })
    return records
