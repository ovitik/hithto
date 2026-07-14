# Reasoning Depth Pilot

Date: 2026-07-12

Goal: run the first small version of the core experiment from the original hypothesis:

- train on Czech reasoning examples with depth 1-3;
- evaluate IID validation and OOD reasoning depth 4-8;
- compare baseline, fixed thought compression, recursive thought compression, and random latent
  controls.

Model:

- `HuggingFaceTB/SmolLM2-360M`
- CPU only
- frozen backbone
- trained parameters: thought builder only

## Important Method Fix

Earlier training/evaluation compressed the full `prompt + answer` sequence. For reasoning-depth
evaluation that is not clean, because latent thoughts could be influenced by the candidate answer.

The new script `scripts/reasoning_depth_experiment.py` builds thought tokens from the prompt only,
then appends candidate answer embeddings and scores answer likelihood. This is slower but much
fairer.

Scoring is forced-choice:

- nesting tasks choose among known locations/containers;
- temporal tasks choose among known event strings;
- predicted answer is the candidate with lowest answer NLL.

## Pilot 1: Replacement, LR 5e-4

Command:

```text
python scripts/reasoning_depth_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants baseline fixed recursive random --seeds 1 --train-n 4 --eval-n 4 --steps 3 --mode replacement --max-prompt-length 112 --max-answer-length 16 --lr 0.0005 --fixed-window 4 --recursive-window 2 --random-latents 42 --out reports/reasoning_depth_pilot_seed1.json
```

| Variant | IID acc | IID gold loss | OOD acc | OOD gold loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.75 | 1.77 | 0.00 | 3.98 |
| fixed replacement | 0.00 | 10.68 | 0.00 | 9.39 |
| recursive replacement | 0.00 | 12.30 | 0.25 | 11.57 |
| random latents | 1.00 | 2.64 | 0.00 | 5.08 |

Interpretation:

- Baseline has useful candidate priors on tiny IID examples but fails OOD depth.
- Replacement thought modules damage prompt information after only 3 CPU steps.
- Recursive gets one OOD example right, but this is not robust evidence.
- Random latents unexpectedly improve tiny IID accuracy, showing that this forced-choice setup is
  sensitive to positional/logit prior effects and needs larger splits.

## Pilot 2: Additive Mode

Command:

```text
python scripts/reasoning_depth_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants fixed recursive --seeds 1 --train-n 4 --eval-n 4 --steps 3 --mode additive --max-prompt-length 112 --max-answer-length 16 --lr 0.0005 --fixed-window 4 --recursive-window 2 --out reports/reasoning_depth_pilot_additive_seed1.json
```

| Variant | IID acc | IID gold loss | OOD acc | OOD gold loss |
| --- | ---: | ---: | ---: | ---: |
| fixed additive | 0.00 | 12.77 | 0.00 | 11.74 |
| recursive additive | 0.00 | 11.18 | 0.25 | 9.72 |

Interpretation:

- Preserving prompt tokens did not fix the issue in this short run.
- Additive recursive is less bad than additive fixed on OOD loss, but still not enough.
- Additive recursive is slower and expands the context instead of compressing it.

## Pilot 3: Replacement, Lower LR 1e-4

Command:

```text
python scripts/reasoning_depth_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants fixed recursive --seeds 1 --train-n 4 --eval-n 4 --steps 3 --mode replacement --max-prompt-length 112 --max-answer-length 16 --lr 0.0001 --fixed-window 4 --recursive-window 2 --out reports/reasoning_depth_pilot_replacement_lr1e4_seed1.json
```

| Variant | IID acc | IID gold loss | OOD acc | OOD gold loss |
| --- | ---: | ---: | ---: | ---: |
| fixed replacement | 0.00 | 7.54 | 0.00 | 8.19 |
| recursive replacement | 0.00 | 10.54 | 0.25 | 9.76 |

Interpretation:

- Lower LR reduced eval loss for fixed replacement but did not recover accuracy.
- Recursive remains worse on loss and only gets one OOD example by candidate selection.

## Current Verdict

This pilot is mostly a negative result for the current naive training setup.

What is confirmed:

- Prompt-only thought scoring works end-to-end.
- Train-depth vs OOD-depth evaluation now exists.
- The baseline model itself fails OOD depth in this tiny forced-choice setup.
- Thought compression can be evaluated without answer leakage.

What is not confirmed:

- Recursive thoughts do not yet improve OOD reasoning depth reliably.
- Fixed/replacement compression currently destroys too much prompt information.
- Additive mode does not automatically help.
- Random latent controls show the metric is noisy on tiny splits.

## Next Experimental Change

The next change should not be "train longer" on this setup. The objective is too unstable.

Better next step:

1. Add reconstruction/distillation losses to prompt-only training.
2. Pretrain fixed compressor to reconstruct prompt hidden states before answer loss.
3. Freeze compressor, train only a small projection/utility head.
4. Then re-enable recursive composition.

The original hypothesis remains testable, but the naive answer-loss-only prompt compressor is not
yet a good learning signal.
