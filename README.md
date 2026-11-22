# Decomposition-Based Learning and Reasoning for Small Language Models

This repository contains the runnable MathGAP study for *Decomposition-Based Learning and Reasoning for Small Language Models*. It compares whole-problem solving with shorter operand calls, citation-based checking, and neural or executable merging. One Qwen3.5-4B adapter trained on five-operation problems serves every trained policy. The paired evaluation uses 1,024 held-out 17-operation problems.

## Related work

[MathGAP](https://github.com/eth-lre/mathgap-experiments) provides the arithmetic proof generator. [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625) motivates sequential subproblems, and [PAL](https://proceedings.mlr.press/v202/gao23f.html) motivates the executable comparison. This study holds weights and ordered source evidence fixed while measuring how splitting an operand, checking its citations, and changing the merge affect different stages of the pipeline.

## Method

The [MathGAP generator](https://github.com/eth-lre/mathgap-experiments/tree/5724e4f65048a62c87da8705c73117a04a4457c3) supplies two independent relation chains per problem. Eighteen shuffled English statements describe the two starting quantities and eight signed updates to each. The question asks for the difference between the endpoints.

1. **Train.** From 512 training problems, build 2,688 supervised examples covering direct solving, planning, child solving, review and merging. Some drafts or children contain controlled errors so that the model can learn to repair them. Fit rank-16 LoRA on query, key, value and output projections of full-attention layers for one epoch (336 updates). The response and its EOS token receive the supervision.
2. **Decompose.** A deterministic controller groups statements by the people they connect and orders each path from its anchor. Qwen identifies the two target holders and their unit. For each target, Qwen solves the first four updates and passes its generated holder, value and unit to a second call for the remaining four. Citation indices map back to the original source statements.
3. **Check and merge.** A deterministic checker validates cited relations and arithmetic against the public source. Nested Full receives the source, both generated workings and checker feedback, then generates the parent answer. Nested Compact shares the same child calls and feedback but passes typed endpoint summaries to the merger. The checker-free ablations omit the feedback from the terminal call. Gold proofs are available during training and gold answers during scoring.
4. **Compare.** Direct generates a single whole-problem solution. Review revises the same trained Direct draft with access to the source and checker. Flat Full solves each entire operand chain in one call. Frozen Direct uses the identical Direct prompt with the adapter disabled.

## Reproduce

Use Python 3.11 or later. From the repository root, create and activate a virtual environment and install the project. With [uv](https://docs.astral.sh/uv/), the following selects a PyTorch build for the host:

```sh
uv venv
uv pip install --torch-backend=auto -c configs/tested-versions.txt -e ".[test]"
```

Activate the environment using `source .venv/bin/activate` on Linux or macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Run the commands below from the repository root. The first model run downloads the pinned Qwen snapshot. CUDA is selected automatically when available, with CPU as the fallback. The study used PyTorch 2.14.1, Transformers 5.20.1, NumPy 2.5.4 and Accelerate 1.15.0. Training and decoding can vary across devices, software versions and batch layouts.

```sh
python -m pytest -q
python -m slm_decomposition.train --config configs/study.yaml --output runs/train
python -m slm_decomposition.infer --config configs/study.yaml --adapter runs/train/adapter.pt --output runs/panel
python -m slm_decomposition.score --config configs/study.yaml --run runs/panel
python -m slm_decomposition.latency --config configs/study.yaml --adapter runs/train/adapter.pt --primary runs/panel --output runs/latency
```

Inference checkpoints completed batches and seals the generated predictions before scoring. Repeat its command to resume. The scorer writes paired accuracy, token counts, batch generation time, endpoint diagnostics and feedback ablations to `runs/panel/scores.json`. The latency replay uses the 24 question IDs and execution-order flags in the study configuration and writes `runs/latency/summary.json`. All model weights, local run records and generated outputs stay outside Git.

Set `evaluation.verified_compact` to `true` in the study configuration to add the verified-carry comparison. Before solving the second chunk, the checker replaces a generated midpoint if the first chunk's own citations establish a different value. Identical prompts reuse their original generations. Use a fresh run directory and the same configuration for inference and scoring.

For a short installation check, add `--max-updates 1` to training in a new output directory. Pass its `smoke-adapter.pt` to inference with `--limit 8` and score that directory. Use a fully trained adapter for accuracy comparisons.

Model size, snapshot, device, training schedule, batch size and token caps are in [the study configuration](configs/study.yaml). For a 2B comparison, change `model.checkpoint` to `Qwen/Qwen3.5-2B` and `model.revision` to `15852e8c16360a2fea060d615a32b45270f8a8fc`. Train a new adapter and use separate output directories. The results below use 4B. To run inference on your own compatible problems, supply `--inputs questions.json` with an array of objects containing only `id`, `document` and `question`. The included scoring key covers the [fixed study panel](data/README.md).

## Reference results

The table reports the archived final adapter (SHA-256 `53a55341feba3fa580a1142ddf95db716c136e223a9b119784c784ed8c7de457`). A fresh run reproduced the training examples, token totals and 336 optimizer updates. Its learned weights differed numerically. To replay an existing adapter, skip training and pass that checkpoint to inference and latency.

Each row charges the complete policy, including shared upstream calls. Batch seconds measure synchronized model generation at batch size eight. They exclude tokenization, source checks and initialization. The warm latency column includes prompt preparation, source checking, generation and decoding on the fixed 24-question subset. Timing is hardware dependent.

| Policy | Correct / 1,024 | Input tokens | Generated tokens | Batch seconds | Warm mean s/question |
|---|---:|---:|---:|---:|---:|
| Frozen Direct | 433 | 660,843 | 512,963 | 2,316.42 | |
| Trained Direct | 852 | 660,843 | 360,541 | 1,023.14 | 6.29 |
| Whole Review | 825 | 1,724,385 | 722,526 | 2,214.13 | 12.64 |
| Flat Full | 912 | 2,337,996 | 417,466 | 1,554.15 | |
| Nested Compact | 1,012 | 2,235,446 | 435,806 | 1,402.48 | 7.82 |
| Nested Full | 1,007 | 2,614,260 | 436,705 | 1,474.36 | 7.92 |
| Review without checker feedback | 812 | 1,705,779 | 723,605 | 2,192.52 | |
| Nested Full without checker feedback | 987 | 2,576,207 | 437,479 | 1,507.95 | |

Nested Full recovered 195 questions missed by Whole Review and lost 13 that Review solved (+17.77 percentage points, paired 95% bootstrap interval +15.23 to +20.31). Compact scored five more correct answers than Full. These results support splitting long chains under this grammar. They leave open whether a merger benefits from retaining full workings. The grammar admits a source-only deterministic solver that answers all 1,024 questions correctly. Every target difference in this panel is positive.
