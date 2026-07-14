# Recursive Thought Token Validation Pass

Date: 2026-07-12

Goal: push beyond integration smoke tests and check whether recursive latent thought tokens show
trainable behavior on a real frozen language model.

Model used for training loops:

- `HuggingFaceTB/SmolLM2-360M`
- loaded as `LlamaForCausalLM`
- CPU only
- training dtype: `fp32`
- backbone frozen
- trained parameters: `ThoughtBuilder`, grouping/compressor/projection heads

## Code Changes

- Recursive interventions now target a `ThoughtNode` that is actually present in the final
  compressed sequence, preferring the highest-level final node.
- Added `scripts/tiny_train_real_model.py` for tiny real-model overfit tests.
- Added `scripts/tiny_seed_sweep.py` for multi-seed fixed-vs-recursive comparison with a single
  model load.
- Added variant-specific window control so recursive models can be compared at the same final token
  budget as one-level fixed pooling.

## Causal Interventions On Final Recursive Nodes

Command:

```text
python scripts/real_gemma_test_matrix.py --config configs/one_level.yaml --model-name HuggingFaceTB/SmolLM2-360M --max-length 128 --n 1 --out reports/real_small_model_test_matrix_final_node.json
```

Result for recursive variant:

- target level: 2
- source positions covered: 0-15
- drop delta: 1.8125
- random replacement delta: 2.9375
- expansion delta: 2.1875

Interpretation: the previous zero-delta recursive result was an artifact of selecting a lower-level
node no longer present in the final sequence. Final recursive thought tokens do causally affect the
logits.

## Tiny Overfit: One Example

Command:

```text
python scripts/tiny_train_real_model.py --model-name HuggingFaceTB/SmolLM2-360M --variant fixed --mode replacement --dtype fp32 --steps 10 --n 1 --max-length 128 --lr 0.001 --out reports/tiny_train_smol360_fixed_replacement_10.json
python scripts/tiny_train_real_model.py --model-name HuggingFaceTB/SmolLM2-360M --variant recursive --mode replacement --dtype fp32 --steps 10 --n 1 --max-length 128 --lr 0.001 --out reports/tiny_train_smol360_recursive_replacement_10.json
```

| Variant | Start answer loss | End answer loss | Delta | Final compression |
| --- | ---: | ---: | ---: | ---: |
| fixed replacement | 11.20 | 5.15 | -6.05 | 3.89x |
| recursive replacement | 11.81 | 5.97 | -5.84 | 14.8x |

Interpretation: both one-level and recursive thought modules are trainable through a frozen real LM.
Recursive tokens can learn even with very aggressive compression, but the result is only an overfit
sanity check.

## Tiny Overfit: Four Examples

Command:

```text
python scripts/tiny_train_real_model.py --model-name HuggingFaceTB/SmolLM2-360M --variant fixed --mode replacement --dtype fp32 --steps 5 --n 4 --max-length 128 --lr 0.0005 --out reports/tiny_train_smol360_fixed_replacement_n4_5.json
python scripts/tiny_train_real_model.py --model-name HuggingFaceTB/SmolLM2-360M --variant recursive --mode replacement --dtype fp32 --steps 5 --n 4 --max-length 128 --lr 0.0005 --out reports/tiny_train_smol360_recursive_replacement_n4_5.json
```

| Variant | Start answer loss | End answer loss | Delta | Final tokens | Compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed replacement | 10.77 | 8.53 | -2.24 | 87 | 3.92x |
| recursive replacement | 11.02 | 3.15 | -7.87 | 23 | 14.83x |

Interpretation: this run is a positive signal for recursion, but it is not enough for a claim. It is
one initialization and one small batch. It does show that recursive compressed latent paths can be
optimized rapidly on real model logits.

## Three-Seed Sweep: Aggressive Recursive Compression

Command:

```text
python scripts/tiny_seed_sweep.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants fixed recursive --seeds 1 2 3 --steps 4 --n 2 --max-length 128 --lr 0.0005 --out reports/tiny_seed_sweep_smol360_fixed_vs_recursive.json
```

| Variant | Mean answer delta | Std | Mean final tokens | Mean compression |
| --- | ---: | ---: | ---: | ---: |
| fixed window=4 | -2.58 | 1.28 | 42.67 | 3.95x |
| recursive window=4 | +0.45 | 2.62 | 11.33 | 14.85x |

Interpretation: recursive window=4 is too aggressive for this early training setup. It compresses
about 15x and is unstable across seeds.

## Three-Seed Sweep: Same Final Token Budget

Command:

```text
python scripts/tiny_seed_sweep.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants fixed recursive --seeds 1 2 3 --steps 4 --n 2 --max-length 128 --lr 0.0005 --fixed-window 4 --recursive-window 2 --out reports/tiny_seed_sweep_smol360_same_compression.json
```

| Variant | Mean answer delta | Std | Mean final tokens | Mean compression |
| --- | ---: | ---: | ---: | ---: |
| fixed window=4 | -2.58 | 1.28 | 42.67 | 3.95x |
| recursive window=2 | -0.90 | 1.19 | 42.67 | 3.95x |

Interpretation: at equal final token budget, recursive tokens are trainable but do not yet beat the
one-level fixed baseline in this short 4-step CPU sweep. This is an important negative/neutral
result: recursion itself is not automatically better. It needs curriculum, less noisy objectives, or
learned boundaries before evaluating OOD reasoning depth.

## Current Evidence

Positive:

- Real model `inputs_embeds` path works for both Gemma 4 E2B and a smaller causal LM.
- Recursive final thought tokens have measurable causal influence on logits.
- Gradients train the thought builder through a frozen real LM.
- Recursive replacement can reduce answer loss on tiny overfit runs.
- Recursive compression can reach much higher compression ratios than one-level fixed pooling.

Negative / unresolved:

- Very aggressive recursion is unstable across seeds.
- At matched final token budget in a short sweep, fixed pooling improved faster than recursive.
- No validation split, OOD depth split, or accuracy-vs-depth result has been run yet.
- No learned-boundary recursive curriculum has been trained for enough steps.

## Next Decision

Do not scale to Gemma 4 training yet. The next research step should be:

1. train fixed pooling until it reliably overfits small batches;
2. train recursive window=2 at matched token budget until stable;
3. only then introduce learned soft/hard boundaries;
4. evaluate IID and OOD-depth splits with at least three seeds.

The current hypothesis remains plausible but unproven. The strongest finding so far is that
recursive latent tokens are technically trainable and causally active, while naive recursion needs a
curriculum to beat simpler fixed pooling.
