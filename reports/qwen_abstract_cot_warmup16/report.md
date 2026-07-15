# Flat Abstract-CoT warm-up gate on RTX 6000 Ada

Run: 2026-07-15 on one Runpod RTX 6000 Ada (48 GB), Qwen/Qwen3-1.7B with 4-bit LoRA.

This was a scaled mechanical test of the flat warm-up idea in *Thinking Without Words*: guided random-code bottleneck SFT followed by constrained prompt-only trace generation and self-distillation. It was not a full reproduction: it used 384 synthetic training examples, 96 held-out examples, a 16-token codebook, 16 abstract-token positions, and no GRPO stage. The planned second policy-improvement round was stopped after the first round failed the trace-information gate.

| Metric after round 1 | Result |
| --- | ---: |
| Answer accuracy | 50/96 (52.1%) |
| Accuracy after trace permutation | 49/96 (51.0%) |
| Distinct generated traces | 1/96 |
| Codebook symbols used | 2/16 |
| Code-token entropy | 0.337 bits |

Every evaluated problem produced the same 16-token sequence: `<AC_09>` followed by fifteen `<AC_14>` tokens. Permuting that sequence made no material difference to answer accuracy. The constrained decoder therefore worked, but the generated trace was a constant and cannot be a reasoning representation.

## Decision

Do not scale this configuration or introduce hierarchy yet. More rounds would train against the same collapsed policy, not test the hypothesis. Before the next GPU run, change the flat objective to make trace content identifiable and prevent constant-code solutions, then validate that a trace permutation causes a substantial answer-accuracy loss. Candidate changes are a stronger mutual-information/anti-collapse objective, curriculum training with shorter traces, and batched cached generation to make the gate economical. Only a flat configuration that passes this causal gate is a valid foundation for a hierarchical variant.

Raw per-example outputs are in `eval_round_1.json`; run metadata and aggregate metrics are in `summary_partial.json`.
