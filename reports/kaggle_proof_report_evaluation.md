# Kaggle Native Recursive Thought Token Proof Evaluation

Date: 2026-07-12

Source report:

```text
recursive_thought_tokens_proof_report.json
```

Config:

- device: CUDA
- train depth: 1-3
- OOD depth: 4-8
- train steps: 1200
- batch size: 256
- entities: 96
- distractor facts: 12
- baseline params: 653,280
- recursive thought params: 193,792

## Main Results

| Model | Params | IID acc depth 1-3 | OOD acc depth 4-8 | IID loss | OOD loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline transformer | 653k | 0.874 | 0.647 | 0.412 | 1.210 |
| recursive thought model | 194k | 0.996 | 0.638 | 0.070 | 2.807 |

## Accuracy By Depth

| Depth | Baseline | Recursive thought | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 0.922 | 1.000 | +0.078 |
| 2 | 0.886 | 0.996 | +0.110 |
| 3 | 0.812 | 0.993 | +0.181 |
| 4 | 0.721 | 0.933 | +0.212 |
| 5 | 0.668 | 0.863 | +0.195 |
| 6 | 0.643 | 0.631 | -0.013 |
| 7 | 0.595 | 0.399 | -0.197 |
| 8 | 0.614 | 0.367 | -0.247 |

## Interpretation

This is the strongest positive result so far for the original idea, but not a full win.

Strongly supportive:

- The recursive thought model has about 30% of the baseline parameters.
- It learns train-depth tasks much better: IID accuracy `0.996` vs `0.874`.
- It strongly outperforms baseline on near-OOD depths 4 and 5.
- It reaches high performance faster during training:
  - at step 360, thought IID `0.946`, baseline IID `0.596`;
  - at step 360, thought OOD `0.626`, baseline OOD `0.420`.
- This supports the core mechanism: a reused latent recursive composition operation can learn a
  reasoning algorithm more parameter-efficiently than a flat transformer on this synthetic task.

Not supportive / unresolved:

- Overall OOD accuracy is slightly lower for recursive thought: `0.638` vs `0.647`.
- OOD loss is much worse for recursive thought: `2.807` vs `1.210`.
- The recursive model collapses at deeper depths:
  - depth 7: `0.399` vs baseline `0.595`;
  - depth 8: `0.367` vs baseline `0.614`.

The pattern is clear:

```text
recursive thought model wins depth 1-5,
roughly ties depth 6,
loses depth 7-8.
```

## What This Means For The Hypothesis

The result supports a moderate version of the hypothesis:

> Native recursive thought tokens are a useful inductive bias for learning compositional reasoning
> from scratch, especially with far fewer parameters and on depths near the training distribution.

It does not yet support the strongest version:

> Recursive thought tokens reliably extrapolate to arbitrarily deeper reasoning.

The model likely learned a useful iterative operation, but it is not stable enough over many
recursive applications. Small errors compound after step 5 or 6.

## Most Likely Failure Mode

The recursive model always applies exactly `MAX_STEPS=8`. It uses a terminal self-loop so the final
answer should remain stable after being reached. The depth 7-8 degradation suggests one of:

- the learned matching/retrieval operation is not sharp enough;
- the state drifts despite the self-loop;
- training on depth 1-3 does not provide enough pressure for stable repeated application;
- the baseline has more parameters and can exploit broader statistical patterns.

## Next Tests

1. Parameter-matched comparison:
   - reduce baseline to about 194k params or increase thought model to about 653k params.

2. Longer-depth curriculum:
   - train 1-3, then gradually include depth 4, then test 5-10.

3. Step-wise probe:
   - evaluate the thought model output after each recursive step, not only after step 8.
   - This will show whether the model reaches the answer and later drifts.

4. Better halting:
   - add learned adaptive stopping or confidence gate instead of always taking exactly 8 steps.

5. Sharper retrieval:
   - add supervised auxiliary loss on which fact should be attended at each step.

6. Multi-seed run:
   - current report is one seed; run at least 3 seeds.

## Verdict

This is a meaningful positive proof-of-principle result. It shows that native recursive thought
tokens can learn a compositional reasoning operation very efficiently. The remaining problem is
long-horizon stability, not basic viability.
