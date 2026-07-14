# Semantic Pretrain To Reasoning Pilot

Date: 2026-07-12

Goal: test the proposed next step:

1. shape thought tokens with semantic losses;
2. fine-tune them on answer likelihood;
3. evaluate IID and OOD reasoning depth.

Script:

```text
scripts/semantic_pretrain_reasoning_experiment.py
```

Model:

- `HuggingFaceTB/SmolLM2-360M`
- frozen backbone
- CPU only
- seed: 1

Command:

```text
python scripts/semantic_pretrain_reasoning_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants fixed recursive --conditions scratch semantic_pretrain --seed 1 --semantic-cases 5 --semantic-train-cases 3 --semantic-steps 3 --answer-steps 2 --train-n 4 --eval-n 4 --semantic-lr 0.0003 --answer-lr 0.0001 --max-prompt-length 112 --max-answer-length 12 --out reports/semantic_pretrain_reasoning_pilot.json
```

## Summary

| Variant | Condition | Semantic margin vs negative | Semantic margin vs counterfactual | IID acc | IID loss | OOD acc | OOD loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | scratch | 0.005 | -0.002 | 0.25 | 8.52 | 0.00 | 10.38 |
| fixed | semantic pretrain | 0.294 | 0.175 | 0.50 | 6.21 | 0.00 | 7.39 |
| recursive | scratch | 0.006 | -0.000 | 0.00 | 10.24 | 0.25 | 9.80 |
| recursive | semantic pretrain | 0.190 | 0.112 | 0.25 | 9.21 | 0.00 | 10.36 |

## Interpretation

Semantic pretraining did what it was supposed to do at the representation level:

- fixed margin vs negative improved from `0.005` to `0.294`;
- fixed margin vs counterfactual improved from `-0.002` to `0.175`;
- recursive margin vs negative improved from `0.006` to `0.190`;
- recursive margin vs counterfactual improved from `-0.000` to `0.112`.

For fixed pooling, semantic pretraining also improved downstream answer evaluation:

- IID accuracy: `0.25 -> 0.50`;
- IID loss: `8.52 -> 6.21`;
- OOD loss: `10.38 -> 7.39`;
- OOD accuracy stayed `0.00`.

For recursive pooling, semantic pretraining improved IID loss and IID accuracy but did not improve
OOD depth:

- IID accuracy: `0.00 -> 0.25`;
- IID loss: `10.24 -> 9.21`;
- OOD accuracy: `0.25 -> 0.00`;
- OOD loss: `9.80 -> 10.36`.

This is a mixed result. It supports the idea that semantic shaping makes thought tokens more useful
representations, but it does not yet show that recursive composition improves OOD reasoning.

## What This Means For The Original Hypothesis

Supported:

- thought tokens can be trained toward meaning-level invariance;
- semantic pretraining can improve answer-likelihood behavior, at least for fixed pooling;
- recursive thoughts can also acquire semantic margins.

Not yet supported:

- recursive thought tokens do not yet reliably improve OOD reasoning depth;
- semantic margins alone are not enough to guarantee useful recursive reasoning;
- the current recursive training/fine-tuning schedule is still unstable.

## Likely Issue

The semantic objective trains the final vector space to separate worlds, but the answer fine-tuning
still has to learn how the frozen LM should consume those vectors as replacement prompt embeddings.
That interface remains poorly calibrated. Fixed pooling benefits more because it is a simpler
one-level distortion. Recursive replacement changes the sequence more aggressively and is harder for
the frozen LM to interpret.

## Next Stronger Test

The next experiment should add explicit interface calibration:

1. semantic pretrain;
2. reconstruction/distillation pretrain against the original prompt hidden states/logits;
3. answer fine-tune;
4. evaluate OOD depth.

In other words, a thought token should be both:

- semantically shaped;
- behaviorally equivalent enough to the original prompt components that the LM can consume it.

The current result says: semantic shaping helps, but recursive thought tokens still need a better
bridge back into the frozen language model.
