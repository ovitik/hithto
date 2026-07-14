# HACoT Gemma 4 E2B Implementation Notes

Date: 2026-07-14

## What Was Added

- `scripts/build_hacot_kaggle_notebook.py`
- `notebooks/kaggle_hacot_gemma4_e2b.ipynb`

The notebook is self-contained and has two modes:

- `HACOT_MODE=SMOKE`: local/control-flow validation without Gemma/JAX.
- `HACOT_MODE=FULL`: hard-requires TPU + JAX/Optax/Orbax + Google DeepMind `gemma`.

## Current External Assumptions

- Gemma 4 E2B exists as a Gemma 4 small effective-parameter model and uses Per-Layer Embeddings.
- The official `gemma` package is the JAX library intended for Gemma use and fine-tuning.
- Runpod is treated as an optional paid fallback. Based on the 2026-07-14 pricing check, the
  default recommendation is Community A100 SXM 80GB, with A100 PCIe 80GB as the cheaper fallback.
  RTX 6000 Ada 48GB or L40S 48GB are suitable for cheap integration/smoke runs, but not as the
  main full-tuning target. H100 is a speed escalation; H200 is memory escalation only.
  The generated notebook writes `reports/runpod_gpu_manifest.json` and requires an explicit
  backend choice instead of silently replacing the Kaggle TPU methodology.

## Data Choice

The notebook prefers public reasoning traces in this order:

1. `ServiceNow-AI/Dolci-Think-SFT`
2. `open-thoughts/OpenThoughts-114k`
3. `NovaSky-AI/Sky-T1_data_17k`
4. `simplescaling/s1K`

If these are unavailable in `SMOKE`, it uses synthetic traces only. In `FULL`, failure to load a
public reasoning source is a hard error. The decontamination report is always written.

## Validation Run

Local validation used `HACOT_MODE=SMOKE` and `HACOT_OUTPUT_DIR=tmp_hacot_smoke`.

Results:

- Notebook JSON generated successfully.
- All code cells parse with `ast.parse`.
- SMOKE execution completed and wrote artifact reports, metrics, tokenizer metadata, manifest, and
  inference wrapper.
- Existing repo tests passed: `13 passed`.

## Important Limitation

The generated notebook currently contains the complete experiment scaffold and the fully tested
grammar/parser/mask/data/report/export path. The low-level FULL Gemma training phase functions are
guarded as TPU-runtime hooks; they should be implemented against the exact installed `gemma`
version on Kaggle after confirming its callable signatures and checkpoint enum names in that
environment. The notebook intentionally does not fake these phases with a local toy backend.
