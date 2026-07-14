# Recursive Thought Tokens

Research prototype for testing whether a language model can create latent, recursively composable
"thought tokens" from its own hidden states.

The first implementation is deliberately a vertical prototype:

```text
Gemma hidden states -> compressor/grouping -> latent thought tokens -> inputs_embeds -> answer loss
```

It includes scaffolding for later phases, but the default scripts are small and local-test friendly.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional QLoRA support:

```bash
pip install -e ".[qlora]"
```

## Quick local smoke test

This does not download Gemma:

```bash
pytest
python scripts/generate_data.py --out data/sample.jsonl --n 20
python scripts/train.py --config configs/one_level.yaml --dry-run
```

## Real Gemma run

The default model id is configurable. The brief requested `google/gemma-4-E2B`; if that checkpoint
requires a newer or model-specific Transformers class, update `configs/*.yaml` and
`src/thought_tokens/gemma_wrapper.py` keeps model loading behind one wrapper.

```bash
python scripts/train.py --config configs/one_level.yaml
python scripts/evaluate.py --config configs/one_level.yaml --checkpoint runs/latest.pt
```

For a frozen-backbone first pass, leave `train.unfreeze_top_layers: 0`. BF16 is the preferred
transparent mode; QLoRA is optional and can be disabled if `inputs_embeds` or hidden-state gradients
become backend-dependent.

## Project Phases

- Phase 0: unchanged Gemma baseline.
- Phase 1: fixed pooling latent tokens.
- Phase 2: learned one-level thought boundaries.
- Phase 3: recursive thought levels.
- Phase 4: adaptive stopping.

Every phase has a config under `configs/` and can log JSONL locally. W&B is intentionally optional.
