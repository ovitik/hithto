# Semantic Thought Token Pilot

Date: 2026-07-12

Goal: test a part of the original idea that was not covered by answer-loss experiments:

- do latent thought tokens become more similar across paraphrases of the same symbolic world?
- do they separate from different worlds?
- do they react to a counterfactual change in the underlying world?
- does recursive composition improve that semantic separation compared with one-level fixed pooling?

Model:

- `HuggingFaceTB/SmolLM2-360M`
- frozen backbone
- CPU only
- trainable module: `ThoughtBuilder`

Script:

```text
scripts/semantic_thought_experiment.py
```

The experiment generates controlled transitive-location worlds:

```text
náramek je v taška. taška je v garáž. garáž je v batoh.
```

Then it creates:

- paraphrase A;
- paraphrase B of the same symbolic chain;
- counterfactual version with a changed final location.

The training objective is intentionally diagnostic:

```text
paraphrase contrastive loss
+ 0.1 reconstruction loss
+ 0.5 counterfactual separation loss
```

This does not yet prove task-solving intelligence, but it directly tests whether the latent thought
space can be shaped toward meaning-level invariance and counterfactual sensitivity.

## Recursive Result

Command:

```text
python scripts/semantic_thought_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variant recursive --cases 5 --train-cases 3 --steps 5 --lr 0.0003 --max-prompt-length 112 --max-answer-length 12 --out reports/semantic_thought_recursive_pilot.json
```

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| same paraphrase cosine | 0.982 | 0.911 | -0.071 |
| negative-world cosine | 0.978 | 0.329 | -0.649 |
| counterfactual cosine | 0.984 | 0.422 | -0.562 |
| margin vs negative | 0.004 | 0.582 | +0.578 |
| margin vs counterfactual | -0.002 | 0.488 | +0.491 |
| compressed gold loss | 10.68 | 10.13 | -0.54 |
| original prompt gold loss | 3.93 | 3.93 | 0.00 |

Interpretation:

- Initially all thought vectors were nearly identical, regardless of meaning.
- After 5 small training steps, paraphrases remain close, while different/counterfactual worlds move
  much farther away.
- This is the strongest evidence so far that the latent thought vectors can be trained toward
  meaning-level units rather than just arbitrary compression vectors.
- The compressed prompt is still much worse than the original prompt for answer likelihood, so this
  is not yet useful reasoning compression.

## Fixed One-Level Result

Command:

```text
python scripts/semantic_thought_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variant fixed --cases 5 --train-cases 3 --steps 5 --lr 0.0003 --max-prompt-length 112 --max-answer-length 12 --out reports/semantic_thought_fixed_pilot.json
```

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| same paraphrase cosine | 0.978 | 0.806 | -0.171 |
| negative-world cosine | 0.973 | 0.431 | -0.543 |
| counterfactual cosine | 0.983 | 0.573 | -0.410 |
| margin vs negative | 0.004 | 0.376 | +0.371 |
| margin vs counterfactual | -0.006 | 0.234 | +0.239 |
| compressed gold loss | 9.84 | 9.85 | +0.02 |
| original prompt gold loss | 3.93 | 3.93 | 0.00 |

Interpretation:

- One-level fixed thought tokens also learn semantic separation.
- Recursive thoughts produce stronger margins in this pilot:
  - recursive margin vs negative: `0.582`
  - fixed margin vs negative: `0.376`
  - recursive margin vs counterfactual: `0.488`
  - fixed margin vs counterfactual: `0.234`
- This is a small but relevant positive signal for recursive composition.

## What This Tests From The Original Hypothesis

This pilot tests:

- paraphrase invariance;
- semantic separation from unrelated worlds;
- sensitivity to counterfactual changes;
- whether higher-level recursive vectors are better semantic summaries than one-level vectors.

It does not yet test:

- long OOD reasoning accuracy;
- reusable thought tokens across many tasks;
- learned boundary quality;
- whether recursive thoughts improve answer generation after full training.

## Current Interpretation

The previous reasoning-depth pilot was mostly negative because answer-loss-only compression destroyed
too much prompt information. This semantic pilot shows a more promising path: direct structure-level
losses can shape latent thoughts into meaning-sensitive units.

The most important result is not that same-paraphrase cosine stayed high; it was high before. The
important result is that negative and counterfactual worlds moved far away while paraphrases remained
comparatively close.

## Next Experiment

Combine this semantic pretraining with answer training:

1. pretrain `ThoughtBuilder` with paraphrase contrastive + reconstruction + counterfactual loss;
2. then fine-tune with prompt-only answer likelihood;
3. evaluate IID/OOD depth again;
4. compare fixed vs recursive at matched token budget.

That is the first fair test of whether meaning-shaped recursive thoughts improve reasoning, rather
than merely being trainable latent vectors.
