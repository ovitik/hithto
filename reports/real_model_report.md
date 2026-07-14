# Real Model Run Report

Date: 2026-07-12

Environment:

- OS/session: Windows PowerShell
- PyTorch: `2.9.1+cpu`
- CUDA: unavailable
- Main model: `google/gemma-4-E2B`
- Fallback model tested: `HuggingFaceTB/SmolLM2-360M`

## What Was Fixed For Real Gemma 4

`google/gemma-4-E2B` is a Gemma 4 multimodal conditional generation model with text
`hidden_size=1536`, not the originally configured `2304`.

Gemma 4 E2B also uses per-layer input embeddings. A plain arbitrary `inputs_embeds` pass fails
because Transformers tries to reverse the embeddings back into `input_ids`. The prototype now
passes both:

- latent `inputs_embeds`;
- pooled `per_layer_inputs`.

For source tokens, `per_layer_inputs` come from Gemma. For thought tokens, the same learned span
attention weights are used to pool the per-layer inputs of the child units.

## Verification Commands

```text
pytest -q
13 passed

python scripts/real_gemma_smoke.py --config configs/one_level.yaml --max-length 48 --n 1
passed

python scripts/real_gemma_test_matrix.py --config configs/one_level.yaml --max-length 128 --n 1
passed

python scripts/real_gemma_test_matrix.py --config configs/one_level.yaml --model-name HuggingFaceTB/SmolLM2-360M --max-length 128 --n 1 --out reports/real_small_model_test_matrix.json
passed
```

## Gemma 4 E2B Results

Model loaded successfully on CPU in `6.62s`.

| Variant | Forward seconds | Answer loss | Thought tokens | Compression |
| --- | ---: | ---: | ---: | --- |
| baseline | 9.14 | 2.47 | 0 | n/a |
| fixed | 10.96 | 19.51 | 13 | 3.92x |
| soft | 15.04 | 36.21 | 9 | 1.28x |
| hard | 17.77 | 14.59 | 4 | 1.11x |
| recursive | 10.37 | 21.23 | 16 | 3.92x, 12.75x |

Causal intervention logit deltas, mean absolute difference:

| Variant | Drop | Random replace | Expand |
| --- | ---: | ---: | ---: |
| fixed | 6.19 | 5.91 | 10.56 |
| soft | 9.31 | 7.53 | 11.31 |
| hard | 3.22 | 1.56 | 4.63 |
| recursive | 0.00 | 0.00 | 0.00 |

Interpretation:

- The real Gemma 4 E2B forward path works with latent thought tokens.
- The untrained thought modules degrade answer loss, which is expected.
- Drop/random/expand interventions change logits for one-level variants, so the latent tokens have
  measurable causal influence on the computation.
- Recursive intervention delta was zero in this tiny untrained run because the final sequence was
  compressed to only a few tokens and the selected first lower-level node was no longer present in
  the final materialized sequence. The intervention script should next target final-level nodes.

## Smaller Fallback Model

`google/gemma-3-1b-pt` and `google/gemma-3-1b-it` exist, but both are manually gated in the current
unauthenticated environment. As a practical non-gated fallback, I tested `HuggingFaceTB/SmolLM2-360M`.

Model loaded successfully on CPU in `26.35s` on first download/load.

| Variant | Forward seconds | Answer loss | Thought tokens | Compression |
| --- | ---: | ---: | ---: | --- |
| baseline | 2.70 | 2.06 | 0 | n/a |
| fixed | 3.38 | 12.64 | 18 | 3.84x |
| soft | 5.21 | 15.07 | 1 | 1.01x |
| hard | 3.49 | 11.68 | 16 | 3.65x |
| recursive | 2.94 | 12.46 | 23 | 3.84x, 14.60x |

Causal intervention logit deltas:

| Variant | Drop | Random replace | Expand |
| --- | ---: | ---: | ---: |
| fixed | 3.86 | 3.61 | 3.89 |
| soft | 0.25 | 0.19 | 0.32 |
| hard | 2.84 | 1.51 | 3.70 |
| recursive | 0.00 | 0.00 | 0.00 |

## Limits Of This Run

This was a real-model integration and intervention smoke test, not a successful research training
run. No meaningful accuracy/generalization claim can be made yet because:

- the backbone was frozen;
- the thought modules were randomly initialized;
- no training was run on the Czech synthetic dataset;
- only one example was used for the CPU matrix;
- no three-seed comparison was run;
- no GPU BF16 or QLoRA training was available in this environment.

## Recommended Next Step

On this CPU-only machine, use `HuggingFaceTB/SmolLM2-360M` or another small non-gated causal LM for
debugging training loops. Use `google/gemma-4-E2B` for final architecture compatibility checks, but
real training should run on a CUDA machine with enough VRAM.

The next implementation fix should make interventions target final-level recursive thoughts, then
run a tiny overfit experiment on 8-16 generated Czech examples with the smaller model.
