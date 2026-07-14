from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_PODS_URL = "https://rest.runpod.io/v1/pods"
DEFAULT_GPU = "NVIDIA RTX 6000 Ada Generation"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_mutation(args: argparse.Namespace) -> str:
    env = [
        {"key": "JUPYTER_PASSWORD", "value": args.jupyter_password},
        {"key": "RUNPOD_REPO_URL", "value": args.repo_url or ""},
        {"key": "QWEN_MODEL_NAME", "value": args.model_name},
        {"key": "QWEN_STEPS", "value": str(args.steps)},
        {"key": "QWEN_TRAIN_N", "value": str(args.train_n)},
        {"key": "QWEN_DEV_N", "value": str(args.dev_n)},
        {"key": "QWEN_EVAL_N", "value": str(args.eval_n)},
        {"key": "QWEN_BATCH_SIZE", "value": str(args.batch_size)},
        {"key": "QWEN_GRAD_ACCUM", "value": str(args.grad_accum)},
    ]
    env_graphql = "[" + ", ".join(
        "{ key: " + json.dumps(item["key"]) + ", value: " + json.dumps(item["value"]) + " }"
        for item in env
    ) + "]"
    return f"""
mutation {{
  podFindAndDeployOnDemand(
    input: {{
      cloudType: COMMUNITY
      gpuCount: 1
      volumeInGb: {args.volume_gb}
      containerDiskInGb: {args.container_disk_gb}
      minVcpuCount: {args.min_vcpu}
      minMemoryInGb: {args.min_ram_gb}
      gpuTypeId: {json.dumps(args.gpu_type)}
      name: {json.dumps(args.name)}
      imageName: {json.dumps(args.image)}
      dockerArgs: ""
      ports: "8888/http"
      volumeMountPath: "/workspace"
      env: {env_graphql}
    }}
  ) {{
    id
    name
    imageName
    machineId
    desiredStatus
    machine {{ podHostId }}
  }}
}}
""".strip()


def post_graphql(api_key: str, query: str) -> dict:
    payload = json.dumps({"query": query}).encode("utf-8")
    url = GRAPHQL_URL + "?api_key=" + urllib.parse.quote(api_key)
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "content-type": "application/json",
            "user-agent": "llm-recursive-tokens-qwen-hacot/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod API HTTP {exc.code}: {body}") from exc


def build_rest_payload(args: argparse.Namespace) -> dict:
    start_cmd = []
    if args.auto_run:
        bootstrap = (
            "set -euo pipefail; "
            "cd /workspace; "
            "if [ ! -d llm-recursive-tokens ]; then git clone \"$RUNPOD_REPO_URL\" llm-recursive-tokens; fi; "
            "cd /workspace/llm-recursive-tokens; "
            "bash scripts/runpod_qwen_bootstrap.sh 2>&1 | tee /workspace/qwen_hacot_bootstrap.log"
        )
        start_cmd = ["bash", "-lc", bootstrap]
    return {
        "allowedCudaVersions": ["12.8", "12.7", "12.6", "12.5", "12.4", "12.1"],
        "cloudType": "COMMUNITY",
        "computeType": "GPU",
        "containerDiskInGb": args.container_disk_gb,
        "cpuFlavorIds": [],
        "cpuFlavorPriority": "availability",
        "dataCenterIds": [],
        "dataCenterPriority": "availability",
        "dockerEntrypoint": [],
        "dockerStartCmd": start_cmd,
        "env": {
            "JUPYTER_PASSWORD": args.jupyter_password,
            "RUNPOD_REPO_URL": args.repo_url or "",
            "QWEN_MODEL_NAME": args.model_name,
            "QWEN_STEPS": str(args.steps),
            "QWEN_TRAIN_N": str(args.train_n),
            "QWEN_DEV_N": str(args.dev_n),
            "QWEN_EVAL_N": str(args.eval_n),
            "QWEN_BATCH_SIZE": str(args.batch_size),
            "QWEN_GRAD_ACCUM": str(args.grad_accum),
        },
        "gpuCount": 1,
        "gpuTypeIds": [args.gpu_type],
        "gpuTypePriority": "availability",
        "imageName": args.image,
        "interruptible": False,
        "locked": False,
        "minRAMPerGPU": max(8, args.min_ram_gb),
        "minVCPUPerGPU": max(2, args.min_vcpu),
        "name": args.name,
        "ports": ["8888/http", "22/tcp"],
        "supportPublicIp": True,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": "/workspace",
    }


def post_rest(api_key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        REST_PODS_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "user-agent": "llm-recursive-tokens-qwen-hacot/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod REST API HTTP {exc.code}: {body_text}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Runpod Community pod for the Qwen HACoT pilot.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api", choices=["rest", "graphql"], default="rest")
    parser.add_argument("--create", action="store_true", help="Actually call the Runpod API. Without this, only prints a dry-run payload.")
    parser.add_argument("--name", default="qwen-hacot-rtx6000ada")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--repo-url", default=os.environ.get("RUNPOD_REPO_URL", ""))
    parser.add_argument("--jupyter-password", default=os.environ.get("RUNPOD_JUPYTER_PASSWORD", "hacot-change-me"))
    parser.add_argument("--volume-gb", type=int, default=120)
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument("--min-vcpu", type=int, default=8)
    parser.add_argument("--min-ram-gb", type=int, default=48)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-n", type=int, default=1800)
    parser.add_argument("--dev-n", type=int, default=240)
    parser.add_argument("--eval-n", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--auto-run", action="store_true", help="Start the Qwen HACoT bootstrap automatically when the pod starts.")
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))
    query = build_mutation(args)
    rest_payload = build_rest_payload(args)
    dry_payload = {
        "api": args.api,
        "graphql_url": GRAPHQL_URL,
        "rest_url": REST_PODS_URL,
        "gpu_type": args.gpu_type,
        "image": args.image,
        "name": args.name,
        "volume_gb": args.volume_gb,
        "container_disk_gb": args.container_disk_gb,
        "repo_url_set": bool(args.repo_url),
        "query": query,
        "rest_payload": rest_payload,
    }
    if not args.create:
        print(json.dumps({"dry_run": True, **dry_payload}, indent=2))
        return

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY is missing. Add it to .env or the environment.", file=sys.stderr)
        raise SystemExit(2)
    result = post_rest(api_key, rest_payload) if args.api == "rest" else post_graphql(api_key, query)
    print(json.dumps(result, indent=2))
    if result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
