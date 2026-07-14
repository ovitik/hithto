# Semantic Pretrain + Side-Channel Pilot

Date: 2026-07-12

Goal: test the next retrofit architecture:

```text
semantic pretrain of ThoughtBuilder
-> side-channel memory adapter training
-> IID/OOD depth evaluation
```

This keeps the normal text prompt intact and appends 4 latent memory tokens derived from the thought
hierarchy. It is intended to avoid the main failure mode of replacement compression.

Script:

```text
scripts/semantic_pretrain_side_channel_experiment.py
```

Command:

```text
python scripts/semantic_pretrain_side_channel_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants fixed recursive --conditions scratch semantic_pretrain --seed 1 --semantic-cases 5 --semantic-train-cases 3 --semantic-steps 2 --side-steps 2 --train-n 4 --eval-n 3 --semantic-lr 0.0003 --side-lr 0.0003 --memory-tokens 4 --max-prompt-length 112 --max-answer-length 12 --out reports/semantic_pretrain_side_channel_pilot.json
```

## Results

| Variant | Condition | Semantic margin vs negative | Semantic margin vs counterfactual | IID acc | IID loss | OOD acc | OOD loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | scratch | 0.003 | -0.005 | 0.667 | 2.016 | 0.000 | 3.528 |
| fixed | semantic pretrain | 0.075 | 0.019 | 0.333 | 1.765 | 0.000 | 3.357 |
| recursive | scratch | 0.003 | -0.003 | 0.667 | 3.031 | 0.333 | 4.476 |
| recursive | semantic pretrain | 0.062 | 0.042 | 0.667 | 1.764 | 0.000 | 3.298 |

## Interpretation

Semantic pretraining improved the representation margins for both fixed and recursive memory.

It also improved OOD loss:

- fixed OOD loss: `3.528 -> 3.357`
- recursive OOD loss: `4.476 -> 3.298`

But it did not improve OOD accuracy in this tiny split:

- fixed stayed at `0.000`;
- recursive went from `0.333` to `0.000`.

That single recursive scratch OOD hit is probably not robust; the loss moved in the better direction
after semantic pretraining, but candidate ranking still did not cross the accuracy threshold.

## What This Means

This is modestly supportive of the side-channel direction:

- preserving text prompt avoids catastrophic replacement behavior;
- semantic pretraining improves latent memory quality;
- semantic-pretrained recursive memory gets substantially better OOD loss than recursive scratch.

It is not yet a win for the strong recursive-thought hypothesis:

- no OOD accuracy gain;
- random/fixed controls are still competitive;
- the side-channel adapter training is unstable with only 2 steps and 4 examples.

## Practical Verdict

The best current signal is:

```text
semantic objectives improve thought-token geometry
and can improve downstream loss,
but we still have not shown reliable reasoning improvement.
```

This suggests the idea deserves a larger, better-controlled run, but the current CPU-scale tests are
too small to resolve the main claim.

## Next Stronger Test

To get a clearer answer, the next run should be less tiny:

- 3 seeds;
- 16-32 train examples;
- 16 IID and 16 OOD eval examples;
- 20-50 side-channel training steps;
- semantic pretrain for at least 20 steps;
- report accuracy by depth and confidence intervals.

That likely needs GPU or a much smaller custom model.

For the CPU-only setting, the more decisive next step is a tiny transformer from scratch where
native hierarchical thought tokens are part of the architecture from the beginning. That would test
the core idea without fighting a frozen pretrained LM interface.
