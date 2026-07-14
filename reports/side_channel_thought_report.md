# Side-Channel Thought Token Pilot

Date: 2026-07-12

Goal: test a different architecture for retrofitting thought tokens onto a frozen LLM.

Instead of replacing text tokens with latent thought embeddings, this experiment keeps the original
prompt intact and appends a small latent memory side-channel:

```text
normal prompt embeddings
+ 4 latent memory tokens derived from thought hierarchy
+ candidate answer embeddings
```

This is more compatible with a pretrained LLM because the model still sees the full text prompt.

Script:

```text
scripts/side_channel_thought_experiment.py
```

Command:

```text
python scripts/side_channel_thought_experiment.py --model-name HuggingFaceTB/SmolLM2-360M --dtype fp32 --variants baseline random fixed recursive --seed 1 --train-n 4 --eval-n 3 --steps 3 --lr 0.0003 --memory-tokens 4 --max-prompt-length 112 --max-answer-length 12 --out reports/side_channel_thought_pilot.json
```

Model:

- `HuggingFaceTB/SmolLM2-360M`
- frozen backbone
- CPU only

## Results

| Variant | Train loss start | Train loss end | IID acc | IID loss | OOD acc | OOD loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | n/a | n/a | 0.667 | 1.924 | 0.000 | 4.021 |
| random memory | 2.352 | 2.125 | 0.667 | 1.412 | 0.000 | 3.264 |
| fixed thought memory | 2.698 | 2.312 | 0.333 | 1.463 | 0.000 | 3.108 |
| recursive thought memory | 2.355 | 2.886 | 0.667 | 2.661 | 0.000 | 4.097 |

## Interpretation

This alternative architecture helps with one earlier problem: the prompt is no longer destroyed by
replacement compression. Losses are much closer to baseline than in replacement mode.

However, this short pilot does **not** support recursive thought memory yet:

- recursive side-channel did not improve OOD accuracy;
- recursive side-channel had worse OOD loss than baseline;
- random learned memory and fixed thought memory improved OOD loss more than recursive memory.

This suggests the issue is not only "the frozen LLM cannot read replacement embeddings." Even when
the original text remains visible, the current recursive memory is not yet a reliably useful
reasoning aid.

## What This Says About The Original Hypothesis

Supported:

- side-channel is a more stable way to attach latent tokens to a pretrained LLM;
- trainable latent memory can reduce answer loss without replacing text;
- fixed thought memory can beat random memory on OOD loss in this tiny run.

Not supported yet:

- recursive memory does not yet outperform fixed or random memory;
- no OOD-depth accuracy gain appeared;
- the current recursive hierarchy is not automatically useful to a frozen pretrained model.

## Likely Cause

The thought hierarchy is still being trained only through a tiny answer-likelihood objective in this
side-channel run. The adapter can learn superficial helpful biases, but the recursive structure has
no strong pressure to encode compositional reasoning unless we add:

- semantic pretraining;
- distillation from original prompt behavior;
- explicit level-wise objectives;
- more examples/seeds.

## Next Test

The strongest next side-channel test would combine:

```text
semantic pretrain
-> side-channel adapter train
-> IID/OOD depth eval
-> random memory control
```

If recursive thought memory still loses after semantic pretraining and adapter calibration, that is
stronger evidence against this retrofit path. If it wins, that would be the first practical support
for the hot-LLM side-channel version of the idea.
