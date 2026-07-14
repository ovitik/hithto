# Phase 1 Vertical Prototype Report

Status: implemented and locally smoke-tested without downloading Gemma.

Verified:

- hidden states are grouped into contiguous spans;
- spans are compressed by a learned attention compressor, not plain averaging;
- latent thought vectors are projected back to the model embedding size;
- replacement and additive sequence composition both run;
- recursive levels can consume lower-level `ThoughtNode` objects;
- gradients flow through the compressor and learned soft boundary module;
- wrapper can rerun a causal LM through `inputs_embeds`;
- causal intervention helpers can drop, replace, shuffle, expand, and ablate thought tokens;
- Czech synthetic data generation works for train and OOD-depth splits.

Local checks:

```text
pytest -q
13 passed

python scripts/train.py --config configs/one_level.yaml --dry-run
{"ok": true, "thought_builder": true}

python scripts/run_ablations.py --hidden-size 16 --seq-len 8
thought_tokens: 4
```

Not yet verified:

- actual `google/gemma-4-E2B` checkpoint download/load;
- BF16/QLoRA behavior on a GPU setup;
- real accuracy, latency, FLOPs, and OOD generalization curves;
- three-seed experimental report.

Decision:

The vertical mechanism is coherent enough to proceed to a small real-model Phase 1 run. The next
checkpoint should use a tiny generated dataset and frozen backbone to confirm that
`hidden states -> latent token -> inputs_embeds -> answer loss` trains with the target Gemma
checkpoint in the available hardware environment.
