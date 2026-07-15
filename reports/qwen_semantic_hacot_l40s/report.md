# Semantic HACoT pilot on L40S

Run: 2026-07-15 on one Runpod L40S, Qwen/Qwen3-1.7B with 4-bit LoRA.

The training set contained 384 synthetic binary programs of depths 1-6. The OOD test contained 96 programs of depths 6-12. Arithmetic was evaluated modulo 10 and boolean programs used AND, OR, and XOR. The variants were direct answer, verbal chain of thought (CoT), a flat abstract program, and a HACoT postfix tree. Every abstract target encoded only operands and operators visible in the question; it did not include intermediate states or the final answer.

| Variant | Answer accuracy | Valid/exact abstract trace | Joint answer + exact trace |
| --- | ---: | ---: | ---: |
| Direct | 42/96 (43.8%) | n/a | 42/96 (43.8%) |
| Verbal CoT, 160 generated tokens | 96/96 (100.0%) | n/a | 96/96 (100.0%) |
| Flat abstract tokens | 37/96 (38.5%) | 0/96 | 0/96 |
| HACoT | 43/96 (44.8%) | 0/96 | 0/96 |

The first CoT evaluation used a 64-token generation limit and measured 7/96, because the answer line was often truncated. Re-evaluation of the saved adapter at 160 tokens gave 96/96 and is the valid result above.

## Conclusion

This pilot does not validate the current abstract-token approach. HACoT's one-answer advantage over direct generation is not meaningful, and the model emitted no valid HACoT forest at all; it generated Qwen internal `<|fim_suffix|>` tokens before often producing an answer. The apparent answer accuracy therefore bypassed the intended abstract program.

The useful positive control is verbal CoT: Qwen learned the same length-generalization task perfectly when given explicit intermediate states. A follow-up should change the representation/training mechanism, not simply run a larger budget: use ordinary token IDs or a separate decoder head for thought symbols, constrain decoding with the HACoT grammar, and require exact program emission before scoring an answer. The evaluator now records answer accuracy, trace validity, exact trace recovery, and joint accuracy to enforce that requirement.
