# Qwen3-1.7B HACoT Pilot On Runpod

Target machine:

- Runpod Community Cloud
- GPU: `NVIDIA RTX 6000 Ada Generation` (Runpod lists this as 48 GB; "46 GB" in the UI usually means usable/free memory after overhead)
- Model: `Qwen/Qwen3-1.7B`
- Method: QLoRA pilot, matched `flat` vs `hacot`

Observed API status on 2026-07-14:

- API key authentication works.
- `RTX 6000 Ada`, `L40S`, `RTX A6000`, and `A40` returned no rentable instances at the time of the check.
- `RTX 4090` reached billing validation, then failed because account balance was too low.
- Therefore the next action is to add Runpod credit, then retry either `RTX 6000 Ada` or the cheaper `RTX 4090` fallback.

## 1. Local Dry Run

```powershell
python scripts\qwen_hacot_pilot.py --dry-run --out-dir runs\qwen_hacot_dryrun
python scripts\runpod_create_qwen_pod.py
```

The second command prints the Runpod GraphQL payload without using the API key.

## 2. Add Secrets

Add to `.env`:

```text
RUNPOD_API_KEY=...
RUNPOD_JUPYTER_PASSWORD=choose-a-password
RUNPOD_REPO_URL=https://github.com/<owner>/<repo>.git
```

If the repo is not reachable from Runpod, create the pod and upload/clone the files manually into:

```text
/workspace/llm-recursive-tokens
```

## 3. Create Pod

```powershell
python scripts\runpod_create_qwen_pod.py --create
```

If the default image is unavailable in Runpod, create the same GPU manually in the UI with a recent
Runpod PyTorch CUDA image, then run the bootstrap command below in the web terminal.

Fallback after adding credit:

```powershell
python scripts\runpod_create_qwen_pod.py --create --api rest --gpu-type "NVIDIA GeForce RTX 4090" --name qwen-hacot-rtx4090 --min-vcpu 2 --min-ram-gb 8 --volume-gb 40 --container-disk-gb 50 --steps 250 --train-n 900 --dev-n 120 --eval-n 60 --batch-size 1 --grad-accum 16
```

This is cheaper and should be enough for a first Qwen3-1.7B QLoRA sanity run. Use it to verify the
pipeline before waiting for a 48GB Ada card.

## 4. Run On Pod

Inside the pod:

```bash
cd /workspace/llm-recursive-tokens
bash scripts/runpod_qwen_bootstrap.sh
```

Main outputs:

```text
/workspace/hacot_runs/qwen_hacot_pilot/summary.json
/workspace/hacot_runs/qwen_hacot_pilot/per_example_results.json
/workspace/hacot_runs/qwen_hacot_pilot/checkpoints/flat
/workspace/hacot_runs/qwen_hacot_pilot/checkpoints/hacot
```

## 5. Budget Guard

First run should stay small:

```bash
QWEN_STEPS=200 QWEN_TRAIN_N=800 QWEN_DEV_N=120 QWEN_EVAL_N=60 bash scripts/runpod_qwen_bootstrap.sh
```

If the first run looks healthy, use the default 600-step run. Escalate only if `hacot` and `flat`
both learn the format and OOD depth accuracy is non-trivial.

## Notes

This is not the final Gemma 4 E2B TPU experiment. It is a cheap falsification/triage run. It is
useful if it compares `hacot` and `flat` under matched data, steps, LoRA rank, sequence length, and
eval prompts. It is not enough for the final strong verdict by itself.
