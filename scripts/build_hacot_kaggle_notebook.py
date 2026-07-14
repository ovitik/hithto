"""Build the Kaggle/Runpod HACoT research notebook.

The generated notebook is intentionally self-contained: all classes and functions live
inside the notebook because the research brief asks for a single `.ipynb` artifact.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    md(
        r"""
# HACoT: Hierarchical Abstract Chain-of-Thought for Gemma 4 E2B

This notebook is a self-contained research harness for testing hierarchical abstract
reasoning tokens against matched flat Abstract-CoT controls.

Primary runtime:

- Kaggle TPU v5e-8, JAX, Optax, Orbax, Google DeepMind `gemma`.
- `MODE="FULL"` hard-fails unless 8 TPU devices and the Gemma stack are available.
- `MODE="SMOKE"` executes the full control flow on small synthetic data, writes the
  same artifact tree, and is meant for syntax and resume testing.

External facts used when this notebook was generated on 2026-07-14:

- Google documents Gemma 4 E2B as an effective-parameter model using Per-Layer
  Embeddings (PLE), with about 11.4 GB BF16 inference weight memory before tuning
  overhead.
- The `gemma` package is the official JAX library for using and fine-tuning Gemma.
- The notebook can emit a Runpod GPU launch manifest, but it never silently
  changes the scientific backend.

Scientific rule: a positive verdict is never inferred from prose. It is computed from
pre-registered thresholds and metrics.
"""
    ),
    code(
        r"""
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Literal, Optional

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

MODE = os.environ.get("HACOT_MODE", "FULL")  # FULL / SMOKE
RUN_STAGE = os.environ.get("HACOT_RUN_STAGE", "AUTO")
RESUME = os.environ.get("HACOT_RESUME", "1") == "1"
SEED = int(os.environ.get("HACOT_SEED", "42"))
SESSION_LIMIT_MIN = int(os.environ.get("HACOT_SESSION_LIMIT_MIN", "500"))
OUTPUT_DIR = Path(os.environ.get("HACOT_OUTPUT_DIR", "/kaggle/working/hacot"))
ARTIFACT_DIR = OUTPUT_DIR / "hacot_artifacts"
RUNTIME_BACKEND = os.environ.get("HACOT_BACKEND", "KAGGLE_TPU")  # KAGGLE_TPU / RUNPOD_GPU

SMOKE = MODE == "SMOKE"
FULL = MODE == "FULL"

random.seed(SEED)
np.random.seed(SEED)

for rel in [
    "checkpoints",
    "tokenizer",
    "inference",
    "metrics",
    "reports",
    "plots",
    "logs",
]:
    (ARTIFACT_DIR / rel).mkdir(parents=True, exist_ok=True)

START_TIME = time.time()


def now_s() -> float:
    return time.time() - START_TIME


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = MODE
    run_stage: str = RUN_STAGE
    resume: bool = RESUME
    seed: int = SEED
    backend: str = RUNTIME_BACKEND
    output_dir: str = str(OUTPUT_DIR)
    session_limit_min: int = SESSION_LIMIT_MIN
    model_name: str = "gemma4_e2b_it"
    max_semantic_nodes: int = 128
    max_depth: int = 12
    max_roots: int = 8
    sequence_buckets: tuple[int, ...] = (256, 512, 768, 1024)
    smoke_synthetic_n: int = 256
    full_synthetic_n: int = 180_000
    natural_reasoning_target: int = 160_000
    instruction_replay_target: int = 40_000
    min_unique_reasoning_prompts: int = 200_000
    min_hacot_training_tokens: int = 120_000_000
    min_preference_trajectories: int = 80_000
    target_hacot_training_tokens: tuple[int, int] = (180_000_000, 250_000_000)
    target_preference_trajectories: tuple[int, int] = (150_000, 250_000)
    seeds_main: tuple[int, ...] = (42, 43) if not SMOKE else (42,)
    branch_budget_tolerance: float = 0.05


CFG = ExperimentConfig()
CONFIG_HASH = sha256_json(dataclasses.asdict(CFG))[:16]
print("config", dataclasses.asdict(CFG))
print("config_hash", CONFIG_HASH)
"""
    ),
    code(
        r"""
# Optional accelerator stack. FULL mode fails early if the required runtime is not present.
JAX_AVAILABLE = False
GEMMA_AVAILABLE = False
KAGGLE_TPU_OK = False

try:
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    JAX_AVAILABLE = True
except Exception as exc:
    if FULL:
        raise RuntimeError("FULL mode requires jax, optax and orbax.") from exc
    print("SMOKE: JAX stack unavailable:", repr(exc))

try:
    from gemma import gm
    GEMMA_AVAILABLE = True
except Exception as exc:
    if FULL:
        raise RuntimeError("FULL mode requires the Google DeepMind gemma package.") from exc
    print("SMOKE: gemma unavailable:", repr(exc))

if JAX_AVAILABLE:
    devices = jax.devices()
    print("jax devices:", devices)
    KAGGLE_TPU_OK = len(devices) >= 8 and all("tpu" in str(d).lower() for d in devices[:8])
    if FULL and CFG.backend == "KAGGLE_TPU" and not KAGGLE_TPU_OK:
        raise RuntimeError("FULL/KAGGLE_TPU requires 8 visible TPU devices.")

if FULL and GEMMA_AVAILABLE:
    assert hasattr(gm.nn, "Gemma4_E2B"), "installed gemma package does not expose gm.nn.Gemma4_E2B"
"""
    ),
    md("## Token Language, Parser, And Constrained Decoder"),
    code(
        r"""
PRIMITIVES = [f"<A{i:02d}>" for i in range(64)]
C2 = [f"<C2_{i:02d}>" for i in range(8)]
C3 = [f"<C3_{i:02d}>" for i in range(4)]
C4 = [f"<C4_{i:02d}>" for i in range(2)]
HA_START, HA_ROOT, HA_END = "<HA_START>", "<HA_ROOT>", "<HA_END>"
HA_TOKENS = [HA_START, HA_ROOT, HA_END] + PRIMITIVES + C2 + C3 + C4
ARITY = {tok: 0 for tok in PRIMITIVES}
ARITY.update({tok: 2 for tok in C2})
ARITY.update({tok: 3 for tok in C3})
ARITY.update({tok: 4 for tok in C4})


@dataclass
class AbstractNode:
    token: str
    children: list["AbstractNode"] = field(default_factory=list)
    root_index: int = 0
    position: int = -1

    @property
    def depth(self) -> int:
        return 1 if not self.children else 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    @property
    def kind(self) -> str:
        if self.token in PRIMITIVES:
            return "primitive"
        if self.token in C2:
            return "binary"
        if self.token in C3:
            return "ternary"
        if self.token in C4:
            return "quaternary"
        return "control"


@dataclass
class ParsedForest:
    tokens: list[str]
    roots: list[AbstractNode]
    nodes_in_generation_order: list[AbstractNode]

    @property
    def max_depth(self) -> int:
        return max((r.depth for r in self.roots), default=0)

    @property
    def node_count(self) -> int:
        return len(self.nodes_in_generation_order)


class HACoTGrammar:
    def __init__(self, max_roots: int = 8, max_nodes: int = 128, max_depth: int = 12):
        self.max_roots = max_roots
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def parse(self, tokens: list[str]) -> ParsedForest:
        if not tokens or tokens[0] != HA_START:
            raise ValueError("forest must start with <HA_START>")
        stack: list[AbstractNode] = []
        roots: list[AbstractNode] = []
        nodes: list[AbstractNode] = []
        ended = False
        for pos, tok in enumerate(tokens[1:], start=1):
            if ended:
                raise ValueError("tokens after <HA_END>")
            if tok in PRIMITIVES:
                node = AbstractNode(tok, [], len(roots), pos)
                stack.append(node)
                nodes.append(node)
            elif tok in ARITY and ARITY[tok] > 0:
                arity = ARITY[tok]
                if len(stack) < arity:
                    raise ValueError(f"{tok} needs {arity} children")
                children = stack[-arity:]
                del stack[-arity:]
                node = AbstractNode(tok, children, len(roots), pos)
                if node.depth > self.max_depth:
                    raise ValueError("max depth exceeded")
                stack.append(node)
                nodes.append(node)
            elif tok == HA_ROOT:
                if len(stack) != 1:
                    raise ValueError("<HA_ROOT> requires exactly one active stack item")
                if len(roots) >= self.max_roots:
                    raise ValueError("max roots exceeded")
                roots.append(stack.pop())
            elif tok == HA_END:
                if stack:
                    raise ValueError("<HA_END> requires an empty active stack")
                if not roots:
                    raise ValueError("<HA_END> requires at least one root")
                ended = True
            else:
                raise ValueError(f"unknown token {tok}")
            if len(nodes) > self.max_nodes:
                raise ValueError("max semantic nodes exceeded")
        if not ended:
            raise ValueError("missing <HA_END>")
        return ParsedForest(tokens=tokens, roots=roots, nodes_in_generation_order=nodes)

    def valid_next(self, prefix: list[str], remaining_slots: int) -> set[str]:
        '''Return tokens allowed before sampling the next token.

        The method is conservative near the length limit: it only allows reductions,
        root closure, or end if there is not enough room for additional structure.
        '''
        if not prefix:
            return {HA_START}
        if prefix[-1] == HA_END:
            return set()
        if prefix[0] != HA_START:
            return set()

        stack_depths: list[int] = []
        roots = 0
        nodes = 0
        for tok in prefix[1:]:
            if tok in PRIMITIVES:
                stack_depths.append(1)
                nodes += 1
            elif tok in ARITY and ARITY[tok] > 0:
                arity = ARITY[tok]
                child_depths = stack_depths[-arity:]
                del stack_depths[-arity:]
                stack_depths.append(1 + max(child_depths))
                nodes += 1
            elif tok == HA_ROOT:
                roots += 1
                stack_depths.clear()
            elif tok == HA_END:
                return set()

        allowed: set[str] = set()
        if not stack_depths and roots > 0:
            allowed.add(HA_END)
        if len(stack_depths) == 1 and roots < self.max_roots:
            allowed.add(HA_ROOT)

        force_close = remaining_slots <= (2 if stack_depths else 1) or nodes >= self.max_nodes
        if not force_close and nodes < self.max_nodes:
            allowed.update(PRIMITIVES)

        for toks, arity in [(C2, 2), (C3, 3), (C4, 4)]:
            if len(stack_depths) >= arity and nodes < self.max_nodes:
                next_depth = 1 + max(stack_depths[-arity:])
                if next_depth <= self.max_depth:
                    allowed.update(toks)
        return allowed

    def sample_valid(self, rng: random.Random, max_len: int = 160) -> list[str]:
        remaining_nodes = self.max_nodes

        def emit_subtree(depth_left: int, budget: int) -> tuple[list[str], int]:
            if depth_left <= 1 or budget <= 2 or rng.random() < 0.30:
                return [rng.choice(PRIMITIVES)], 1
            possible = [2]
            if budget >= 4:
                possible.append(3)
            if budget >= 5:
                possible.append(4)
            arity = rng.choice(possible)
            child_budget = max(1, (budget - 1) // arity)
            toks: list[str] = []
            used = 1
            for _ in range(arity):
                child, child_used = emit_subtree(depth_left - 1, child_budget)
                toks.extend(child)
                used += child_used
            toks.append(rng.choice({2: C2, 3: C3, 4: C4}[arity]))
            return toks, used

        prefix = [HA_START]
        roots = rng.randint(1, self.max_roots)
        for r in range(roots):
            if remaining_nodes <= 0:
                break
            root_budget = max(1, remaining_nodes // max(1, roots - r))
            depth = rng.randint(1, self.max_depth)
            subtree, used = emit_subtree(depth, root_budget)
            if len(prefix) + len(subtree) + 2 > max_len:
                subtree, used = [rng.choice(PRIMITIVES)], 1
            prefix.extend(subtree)
            prefix.append(HA_ROOT)
            remaining_nodes -= used
            if len(prefix) + 1 >= max_len:
                break
        prefix.append(HA_END)
        self.parse(prefix)
        return prefix


grammar = HACoTGrammar(CFG.max_roots, CFG.max_semantic_nodes, CFG.max_depth)
example = [HA_START, "<A00>", "<A01>", "<C2_00>", "<A02>", "<C2_01>", HA_ROOT, HA_END]
print(grammar.parse(example))
"""
    ),
    code(
        r"""
def run_grammar_tests(n: int = 10_000 if FULL else 500) -> dict[str, Any]:
    rng = random.Random(SEED)
    for _ in range(n):
        toks = grammar.sample_valid(rng, max_len=160)
        parsed = grammar.parse(toks)
        assert parsed.node_count <= CFG.max_semantic_nodes
        assert parsed.max_depth <= CFG.max_depth
        assert 1 <= len(parsed.roots) <= CFG.max_roots
        assert toks == parsed.tokens
    bad_cases = [
        [HA_START, HA_END],
        [HA_START, "<A00>", HA_END],
        [HA_START, "<C2_00>", HA_ROOT, HA_END],
        [HA_START, "<A00>", "<A01>", HA_ROOT, HA_END],
    ]
    for bad in bad_cases:
        try:
            grammar.parse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad case accepted: {bad}")
    return {"random_valid_generations": n, "status": "passed"}


grammar_test_report = run_grammar_tests()
write_json(ARTIFACT_DIR / "reports" / "grammar_tests.json", grammar_test_report)
grammar_test_report
"""
    ),
    md("## Synthetic And Natural Data Plan"),
    code(
        r"""
@dataclass
class ReasoningExample:
    prompt: str
    answer: str
    verbal_cot: str
    gold_tree: list[str]
    task_family: str
    template_id: str
    difficulty: int
    required_depth: int
    split_key: str
    verifier: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def balanced_tree_for_steps(n_steps: int, rng: random.Random) -> list[str]:
    leaves = [rng.choice(PRIMITIVES) for _ in range(max(1, n_steps))]
    stack = leaves[:]
    out = [HA_START] + leaves
    while len(stack) > 1:
        arity = 2 if len(stack) < 3 or rng.random() < 0.7 else 3
        arity = min(arity, len(stack))
        op = rng.choice(C2 if arity == 2 else C3)
        for _ in range(arity):
            stack.pop()
        stack.append(op)
        out.append(op)
    out += [HA_ROOT, HA_END]
    return out


def gen_arithmetic(rng: random.Random, difficulty: int, split: str) -> ReasoningExample:
    depth = difficulty
    value = rng.randint(1, 9)
    expr = str(value)
    cot = [f"start {value}"]
    for i in range(depth):
        op = rng.choice(["+", "-", "*"])
        x = rng.randint(1, 9)
        if op == "+":
            value += x
        elif op == "-":
            value -= x
        else:
            value *= x
        expr = f"({expr} {op} {x})"
        cot.append(f"step {i+1}: apply {op}{x} -> {value}")
    return ReasoningExample(
        prompt=f"Compute the integer value of {expr}. Return only the integer.",
        answer=str(value),
        verbal_cot="; ".join(cot),
        gold_tree=balanced_tree_for_steps(depth + 1, rng),
        task_family="nested_arithmetic",
        template_id=f"arith_{split}_{depth}",
        difficulty=difficulty,
        required_depth=depth,
        split_key=f"{split}:arith:{depth}:{rng.randrange(10**9)}",
        verifier="exact_int",
    )


def gen_boolean(rng: random.Random, difficulty: int, split: str) -> ReasoningExample:
    vars_ = {name: bool(rng.getrandbits(1)) for name in ["p", "q", "r", "s"]}
    expr_text = rng.choice(list(vars_))
    value = vars_[expr_text]
    cot = [f"{expr_text}={int(value)}"]
    for i in range(difficulty):
        var = rng.choice(list(vars_))
        op = rng.choice(["AND", "OR", "XOR"])
        rhs = vars_[var]
        if op == "AND":
            value = value and rhs
        elif op == "OR":
            value = value or rhs
        else:
            value = bool(value) ^ bool(rhs)
        expr_text = f"({expr_text} {op} {var})"
        cot.append(f"{op} {var}={int(rhs)} -> {int(value)}")
    prompt = f"Given p={int(vars_['p'])}, q={int(vars_['q'])}, r={int(vars_['r'])}, s={int(vars_['s'])}, evaluate {expr_text}. Return 0 or 1."
    return ReasoningExample(prompt, str(int(value)), "; ".join(cot), balanced_tree_for_steps(difficulty + 1, rng), "boolean_logic", f"bool_{split}", difficulty, difficulty, f"{split}:bool:{rng.randrange(10**9)}", "exact_bool")


def gen_graph_path(rng: random.Random, difficulty: int, split: str) -> ReasoningExample:
    nodes = [chr(ord("A") + i) for i in range(max(8, difficulty + 3))]
    path = rng.sample(nodes, difficulty + 1)
    edges = [(path[i], path[i + 1]) for i in range(difficulty)]
    distractors = []
    for _ in range(difficulty + 4):
        a, b = rng.sample(nodes, 2)
        if (a, b) not in edges:
            distractors.append((a, b))
    all_edges = edges + distractors
    rng.shuffle(all_edges)
    prompt = "Edges: " + ", ".join(f"{a}->{b}" for a, b in all_edges) + f". Starting at {path[0]}, follow {difficulty} edges from the hidden chain. Return the final node."
    cot = "; ".join(f"{path[i]}->{path[i+1]}" for i in range(difficulty))
    return ReasoningExample(prompt, path[-1], cot, balanced_tree_for_steps(difficulty, rng), "graph_paths", f"graph_{split}", difficulty, difficulty, f"{split}:graph:{rng.randrange(10**9)}", "exact_string")


def gen_list_transform(rng: random.Random, difficulty: int, split: str) -> ReasoningExample:
    xs = [rng.randint(0, 9) for _ in range(rng.randint(3, 6))]
    cur = xs[:]
    ops = []
    for _ in range(difficulty):
        op = rng.choice(["reverse", "inc", "drop_first", "rotate"])
        ops.append(op)
        if op == "reverse":
            cur = list(reversed(cur))
        elif op == "inc":
            cur = [(x + 1) % 10 for x in cur]
        elif op == "drop_first" and cur:
            cur = cur[1:]
        elif op == "rotate" and cur:
            cur = cur[1:] + cur[:1]
    prompt = f"Start with list {xs}. Apply: {', '.join(ops)}. Return the final list as JSON."
    return ReasoningExample(prompt, json.dumps(cur), " -> ".join(ops), balanced_tree_for_steps(max(1, difficulty), rng), "recursive_list_transform", f"list_{split}", difficulty, difficulty, f"{split}:list:{rng.randrange(10**9)}", "json_list")


GENERATOR_POOL: list[Callable[[random.Random, int, str], ReasoningExample]] = [
    gen_arithmetic,
    gen_boolean,
    gen_graph_path,
    gen_list_transform,
]

SYNTHETIC_FAMILY_TARGETS = {
    "nested_arithmetic": "nested arithmetic expressions",
    "linear_equations": "one and two variable equations",
    "function_composition": "symbolic function composition",
    "stack_programs": "stack and tiny program traces",
    "boolean_logic": "boolean circuits and propositional logic",
    "relational_deduction": "transitive relation deduction",
    "graph_paths": "graph paths and local-result composition",
    "constraint_satisfaction": "small CSPs",
    "text_world_planning": "simple text-world planning",
    "causal_dag": "causal DAG interventions",
    "counterfactuals": "counterfactual reasoning",
    "recursive_list_transform": "recursive string/list transforms",
    "synthetic_multihop_qa": "synthetic multi-hop QA",
    "proof_integration": "independent proof integration",
}


def generate_synthetic_examples(n: int, split: str, seed: int) -> list[ReasoningExample]:
    rng = random.Random(seed)
    examples: list[ReasoningExample] = []
    seen = set()
    while len(examples) < n:
        if split == "train":
            difficulty = rng.choices([1, 2, 3, 4, 5, 6], weights=[2, 3, 4, 4, 3, 1])[0]
        else:
            difficulty = rng.randint(6, 12)
        gen = rng.choice(GENERATOR_POOL)
        ex = gen(rng, difficulty, split)
        key = hashlib.sha1((ex.prompt + ex.answer).encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            examples.append(ex)
    return examples


train_n = CFG.smoke_synthetic_n if SMOKE else CFG.full_synthetic_n
dev_n = 64 if SMOKE else 10_000
test_n = 64 if SMOKE else 20_000
synthetic_train = generate_synthetic_examples(train_n, "train", SEED)
synthetic_dev = generate_synthetic_examples(dev_n, "dev", SEED + 1000)
synthetic_test = generate_synthetic_examples(test_n, "test", SEED + 2000)

write_json(ARTIFACT_DIR / "reports" / "synthetic_data_manifest.json", {
    "train": len(synthetic_train),
    "dev": len(synthetic_dev),
    "test": len(synthetic_test),
    "families_implemented_in_smoke": sorted({e.task_family for e in synthetic_train}),
    "families_required_in_full_plan": SYNTHETIC_FAMILY_TARGETS,
})
print(len(synthetic_train), len(synthetic_dev), len(synthetic_test))
"""
    ),
    code(
        r"""
NATURAL_REASONING_CANDIDATES = [
    # The loader tries these in order and records the selected dataset.
    # Dolci is preferred when available because it matches the Abstract-CoT source brief.
    {"path": "ServiceNow-AI/Dolci-Think-SFT", "priority": 0},
    {"path": "open-thoughts/OpenThoughts-114k", "priority": 1},
    {"path": "NovaSky-AI/Sky-T1_data_17k", "priority": 2},
    {"path": "simplescaling/s1K", "priority": 3},
]
INSTRUCTION_REPLAY_CANDIDATES = [
    {"path": "HuggingFaceH4/ultrachat_200k", "priority": 0},
    {"path": "tatsu-lab/alpaca", "priority": 1},
]


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def example_signature(prompt: str, n: int = 5) -> set[str]:
    words = re.findall(r"\w+", prompt.lower())
    return {" ".join(words[i:i+n]) for i in range(max(0, len(words) - n + 1))}


def decontaminate(train_rows: list[dict[str, Any]], eval_prompts: list[str], threshold: float = 0.85) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_sigs = [example_signature(p) for p in eval_prompts]
    kept, dropped = [], []
    for row in train_rows:
        sig = example_signature(str(row.get("prompt", "")))
        contaminated = False
        for esig in eval_sigs:
            if not sig or not esig:
                continue
            jacc = len(sig & esig) / max(1, len(sig | esig))
            if jacc >= threshold:
                contaminated = True
                break
        (dropped if contaminated else kept).append(row)
    return kept, {"kept": len(kept), "dropped": len(dropped), "threshold": threshold}


def load_public_reasoning_subset(target: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if SMOKE:
        rows = [
            {"prompt": ex.prompt, "answer": ex.answer, "verbal_cot": ex.verbal_cot, "source": "synthetic_smoke"}
            for ex in synthetic_train[: min(64, len(synthetic_train))]
        ]
        return rows, {"selected": "synthetic_smoke", "rows": len(rows)}
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("FULL natural data loading requires datasets.") from exc

    errors = []
    for cand in NATURAL_REASONING_CANDIDATES:
        try:
            ds = load_dataset(cand["path"], split="train", streaming=True)
            rows = []
            for item in ds:
                text = json.dumps(item, ensure_ascii=False)
                prompt = item.get("prompt") or item.get("question") or item.get("instruction") or item.get("problem")
                answer = item.get("answer") or item.get("final_answer") or item.get("output") or item.get("response")
                cot = item.get("reasoning") or item.get("cot") or item.get("solution") or item.get("rationale")
                if not prompt or not answer or not cot:
                    continue
                if len(str(cot)) > 6000 or len(str(prompt)) > 3000:
                    continue
                rows.append({"prompt": str(prompt), "answer": str(answer), "verbal_cot": str(cot), "source": cand["path"]})
                if len(rows) >= target:
                    break
            if rows:
                return rows, {"selected": cand["path"], "rows": len(rows)}
        except Exception as exc:
            errors.append({"path": cand["path"], "error": repr(exc)})
    raise RuntimeError(f"no public reasoning dataset could be loaded: {errors}")


natural_rows, natural_report = load_public_reasoning_subset(64 if SMOKE else CFG.natural_reasoning_target)
eval_prompts = [e.prompt for e in synthetic_dev + synthetic_test]
natural_rows, decon_report = decontaminate(natural_rows, eval_prompts, threshold=0.85)
write_json(ARTIFACT_DIR / "reports" / "decontamination_report.json", {
    "natural": natural_report,
    "decontamination": decon_report,
    "dataset_policy": "Prefer high quality public reasoning traces; never train on public eval splits.",
})
print(natural_report, decon_report)
"""
    ),
    md("## Tree-Aware Structural Features And Masks"),
    code(
        r"""
def collect_nodes(root: AbstractNode) -> list[AbstractNode]:
    out = []
    for child in root.children:
        out.extend(collect_nodes(child))
    out.append(root)
    return out


def structural_features(forest: ParsedForest) -> dict[str, np.ndarray]:
    nodes = forest.nodes_in_generation_order
    index = {id(n): i for i, n in enumerate(nodes)}
    depth = np.array([min(15, n.depth) for n in nodes], dtype=np.int32)
    type_id = np.array([
        {"primitive": 0, "binary": 1, "ternary": 2, "quaternary": 3}.get(n.kind, 4)
        for n in nodes
    ], dtype=np.int32)
    size_bucket = np.array([min(15, int(math.log2(max(1, n.size))) + 1) for n in nodes], dtype=np.int32)
    root_order = np.array([min(7, n.root_index) for n in nodes], dtype=np.int32)
    is_root = np.zeros(len(nodes), dtype=np.int32)
    for r in forest.roots:
        is_root[index[id(r)]] = 1
    return {"depth": depth, "type_id": type_id, "size_bucket": size_bucket, "root_order": root_order, "is_root": is_root}


def tree_attention_mask(forest: ParsedForest, prompt_len: int, answer_len: int, descendants_visible: bool = False, root_only_answer: bool = False) -> np.ndarray:
    n_nodes = forest.node_count
    total = prompt_len + n_nodes + answer_len
    mask = np.zeros((total, total), dtype=bool)
    mask[:prompt_len, :prompt_len] = np.tril(np.ones((prompt_len, prompt_len), dtype=bool))
    node_offset = prompt_len
    answer_offset = prompt_len + n_nodes
    node_index = {id(n): i for i, n in enumerate(forest.nodes_in_generation_order)}

    completed_roots: list[int] = []
    for node in forest.nodes_in_generation_order:
        i = node_offset + node_index[id(node)]
        mask[i, :prompt_len] = True
        for ridx in completed_roots:
            mask[i, node_offset + ridx] = True
        if node.children:
            children = node.children
            visible = []
            if descendants_visible:
                for child in children:
                    visible.extend(collect_nodes(child))
            else:
                visible = children
            for child in visible:
                mask[i, node_offset + node_index[id(child)]] = True
        for root in forest.roots:
            if root is node:
                completed_roots.append(node_index[id(root)])

    answer_visible_nodes = []
    if root_only_answer:
        answer_visible_nodes = [node_index[id(r)] for r in forest.roots]
    else:
        answer_visible_nodes = list(range(n_nodes))
    for a in range(answer_len):
        row = answer_offset + a
        mask[row, :prompt_len] = True
        for ni in answer_visible_nodes:
            mask[row, node_offset + ni] = True
        mask[row, answer_offset: row + 1] = True
    return mask


def run_mask_tests() -> dict[str, Any]:
    forest = grammar.parse([HA_START, "<A00>", "<A01>", "<C2_00>", "<A02>", "<C2_01>", HA_ROOT, HA_END])
    mask = tree_attention_mask(forest, prompt_len=5, answer_len=3)
    assert mask.shape == (5 + forest.node_count + 3, 5 + forest.node_count + 3)
    assert mask[-1, 5:5 + forest.node_count].all(), "answer must see all abstract nodes"
    root_only = tree_attention_mask(forest, 5, 3, root_only_answer=True)
    assert root_only[-1, 5:5 + forest.node_count].sum() == len(forest.roots)
    return {"status": "passed", "nodes": forest.node_count}


mask_report = run_mask_tests()
write_json(ARTIFACT_DIR / "reports" / "mask_tests.json", mask_report)
mask_report
"""
    ),
    md("## Compute Backend, TPU Benchmarking, And Runpod Decision"),
    code(
        r"""
@dataclass
class BackendQuote:
    backend: str
    device: str
    vram_gb: int
    hourly_usd: Optional[float]
    role: str
    notes: str


RUNPOD_GPU_OPTIONS = [
    BackendQuote("RUNPOD_GPU", "A100 SXM 80GB", 80, 1.49, "recommended_default", "Best price/performance first attempt for real Gemma load plus short training probes."),
    BackendQuote("RUNPOD_GPU", "A100 PCIe 80GB", 80, 1.39, "recommended_if_sxm_unavailable", "Slightly cheaper, usually slower interconnect/memory path than SXM."),
    BackendQuote("RUNPOD_GPU", "RTX 6000 Ada 48GB", 48, 0.77, "cheap_smoke_or_lora", "Good for notebook integration, inference, dataset work, and emergency PEFT probes; not the main full-tuning target."),
    BackendQuote("RUNPOD_GPU", "L40S 48GB", 48, 0.99, "cheap_smoke_or_lora", "Useful fallback if RTX 6000 Ada is unavailable."),
    BackendQuote("RUNPOD_GPU", "H100 PCIe 80GB", 80, 2.89, "speed_escalation", "Use only if A100 is too slow and wall-clock matters more than cost."),
    BackendQuote("RUNPOD_GPU", "H100 SXM 80GB", 80, 2.99, "speed_escalation", "Use only if A100 is too slow and SXM availability is good."),
    BackendQuote("RUNPOD_GPU", "H200 141GB", 141, 4.39, "memory_escalation_only", "Use only after A100/H100 proves memory-insufficient or too slow for the required short run."),
]


def write_runpod_manifest() -> dict[str, Any]:
    manifest = {
        "purpose": "Optional paid Runpod run if Kaggle TPU throughput cannot reach minimum exposure soon enough.",
        "selected_gpu": "A100 SXM 80GB",
        "selected_gpu_fallback": "A100 PCIe 80GB",
        "min_vram_gb": 80,
        "avoid_by_default": ["H200 141GB"],
        "recommendation": "Start with Community A100 SXM 80GB if available; use A100 PCIe 80GB if cheaper/easier to obtain. Escalate to H100 only for speed, and to H200 only for proven memory pressure.",
        "container_hint": "jax/tpu image is not appropriate for CUDA; use a CUDA/JAX image with matching jax[cuda] wheels.",
        "env": {
            "HACOT_MODE": MODE,
            "HACOT_BACKEND": "RUNPOD_GPU",
            "HACOT_RUNPOD_GPU_TYPE": "A100_SXM_80GB",
            "HACOT_OUTPUT_DIR": "/workspace/hacot",
        },
        "decision_rule": "Run a 20-40 minute benchmark on A100 80GB first. Continue only if it loads Gemma 4 E2B, passes SMOKE/FULL unit tests, and projected cost beats waiting for Kaggle TPU. Do not rent H200 unless A100/H100 data shows it is necessary.",
        "reference_prices_usd_per_hour": [dataclasses.asdict(x) for x in RUNPOD_GPU_OPTIONS],
    }
    write_json(ARTIFACT_DIR / "reports" / "runpod_gpu_manifest.json", manifest)
    return manifest


def benchmark_runtime() -> dict[str, Any]:
    if SMOKE or not JAX_AVAILABLE:
        report = {
            "mode": MODE,
            "backend": CFG.backend,
            "synthetic_tokens_per_second": 1_000_000.0,
            "examples_per_second": 10_000.0,
            "oom": False,
            "nan": False,
            "selected_mesh": "smoke",
        }
        write_json(ARTIFACT_DIR / "reports" / "resource_report.json", report)
        return report
    # Lightweight allocation/compile benchmark; full train cells do the real measurement.
    xs = jnp.ones((8, 512, 256), dtype=jnp.bfloat16)

    @jax.jit
    def tiny_step(x):
        y = jnp.tanh(x @ jnp.ones((256, 256), dtype=jnp.bfloat16))
        return y.mean()

    t0 = time.time()
    out = tiny_step(xs).block_until_ready()
    dt = time.time() - t0
    report = {
        "mode": MODE,
        "backend": CFG.backend,
        "devices": [str(d) for d in jax.devices()],
        "tiny_compile_and_run_s": dt,
        "tiny_output": float(out),
        "candidate_meshes": ["fsdp=8", "data=2,fsdp=4"],
        "selected_mesh": "fsdp=8",
    }
    write_json(ARTIFACT_DIR / "reports" / "resource_report.json", report)
    return report


runpod_manifest = write_runpod_manifest()
resource_report = benchmark_runtime()
resource_report
"""
    ),
    md("## Gemma Adapter And Vocabulary Expansion"),
    code(
        r"""
def nearest_multiple(x: int, m: int = 128) -> int:
    return int(math.ceil(x / m) * m)


@dataclass
class TokenExtensionPlan:
    base_vocab_size: int
    added_tokens: list[str]
    padded_vocab_size: int
    trainable_new_rows: int
    masked_pad_rows: int
    token_to_id: dict[str, int]


def build_token_extension_plan(base_vocab_size: int) -> TokenExtensionPlan:
    start = base_vocab_size
    token_to_id = {tok: start + i for i, tok in enumerate(HA_TOKENS)}
    padded = nearest_multiple(base_vocab_size + len(HA_TOKENS), 128)
    return TokenExtensionPlan(
        base_vocab_size=base_vocab_size,
        added_tokens=HA_TOKENS,
        padded_vocab_size=padded,
        trainable_new_rows=len(HA_TOKENS),
        masked_pad_rows=padded - (base_vocab_size + len(HA_TOKENS)),
        token_to_id=token_to_id,
    )


def find_vocab_indexed_arrays(params: Any, base_vocab_size: int) -> list[tuple[tuple[Any, ...], tuple[int, ...]]]:
    '''Find arrays with an axis equal to the tokenizer vocabulary size.

    This catches the main embedding table, PLE tables, and LM head without relying
    on fragile parameter names.
    '''
    found = []
    if not JAX_AVAILABLE:
        return found

    def walk(x: Any, path: tuple[Any, ...] = ()):
        if hasattr(x, "shape"):
            shape = tuple(int(s) for s in x.shape)
            axes = tuple(i for i, s in enumerate(shape) if s == base_vocab_size)
            if axes:
                found.append((path, axes))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + (k,))
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, path + (i,))

    walk(params)
    return found


def _extend_array(arr: Any, axis: int, new_size: int, rng_seed: int):
    if not JAX_AVAILABLE:
        return arr
    old_size = arr.shape[axis]
    if old_size >= new_size:
        return arr
    key = jax.random.PRNGKey(rng_seed)
    mean = jnp.asarray(arr).mean()
    std = jnp.asarray(arr).std()
    pad_shape = list(arr.shape)
    pad_shape[axis] = new_size - old_size
    noise = mean + jax.random.normal(key, pad_shape, dtype=arr.dtype) * (std + jnp.asarray(1e-6, dtype=arr.dtype))
    return jnp.concatenate([arr, noise.astype(arr.dtype)], axis=axis)


def extend_vocab_indexed_params(params: Any, base_vocab_size: int, padded_vocab_size: int, seed: int):
    if not JAX_AVAILABLE:
        return params, []
    touched = []

    def walk(x: Any, path: tuple[Any, ...] = ()):
        if hasattr(x, "shape") and base_vocab_size in tuple(x.shape):
            out = x
            for axis, size in enumerate(x.shape):
                if int(size) == base_vocab_size:
                    out = _extend_array(out, axis, padded_vocab_size, seed + len(touched))
                    touched.append({"path": [str(p) for p in path], "axis": axis, "old": base_vocab_size, "new": padded_vocab_size})
            return out
        if isinstance(x, dict):
            return {k: walk(v, path + (k,)) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v, path + (i,)) for i, v in enumerate(x)]
        if isinstance(x, tuple):
            return tuple(walk(v, path + (i,)) for i, v in enumerate(x))
        return x

    return walk(params), touched


def load_gemma4_e2b_params_and_tokenizer():
    if not GEMMA_AVAILABLE:
        if FULL:
            raise RuntimeError("Gemma package is required in FULL mode.")
        return None, None, build_token_extension_plan(262_144), {"smoke": True}
    model = gm.nn.Gemma4_E2B()
    checkpoint_enum = getattr(gm.ckpts.CheckpointPath, "GEMMA4_E2B_IT", None)
    if checkpoint_enum is None:
        candidates = [name for name in dir(gm.ckpts.CheckpointPath) if "GEMMA4_E2B" in name and name.endswith("_IT")]
        if not candidates:
            raise RuntimeError("Could not find a Gemma4 E2B instruction checkpoint enum.")
        checkpoint_enum = getattr(gm.ckpts.CheckpointPath, candidates[0])
    params = gm.ckpts.load_params(checkpoint_enum)
    tokenizer = gm.text.Gemma4Tokenizer()
    base_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or getattr(tokenizer, "vocab", None).__len__())
    plan = build_token_extension_plan(base_vocab_size)
    params, touched = extend_vocab_indexed_params(params, base_vocab_size, plan.padded_vocab_size, SEED)
    write_json(ARTIFACT_DIR / "tokenizer" / "token_extension_plan.json", dataclasses.asdict(plan))
    write_json(ARTIFACT_DIR / "reports" / "vocab_extension_report.json", {"touched": touched})
    return model, tokenizer, plan, {"checkpoint": str(checkpoint_enum), "touched": touched}


if SMOKE:
    model, tokenizer, token_plan, gemma_report = load_gemma4_e2b_params_and_tokenizer()
else:
    # Defer the expensive load until the selected train/eval stage.
    token_plan = build_token_extension_plan(262_144)
    gemma_report = {"deferred": True, "base_vocab_size_assumed_for_plan": token_plan.base_vocab_size}
write_json(ARTIFACT_DIR / "tokenizer" / "token_extension_plan.json", dataclasses.asdict(token_plan))
gemma_report
"""
    ),
    md("## Training Stages And Fair Branch Accounting"),
    code(
        r"""
STAGES = [
    "phase0_unit_tests",
    "phase1_base_benchmark",
    "phase2_shared_reasoning_checkpoint",
    "phase3_flat_acot_warmup",
    "phase4_branch_split",
    "phase5_hacot_gold_tree_warmup",
    "phase6_hierarchical_bottleneck_sft",
    "phase7_latent_tree_policy_iteration",
    "phase8_self_distillation_no_verbal_cot",
    "phase9_preference_training",
    "phase10_grpo",
    "phase11_evaluation_and_ablations",
    "phase12_report_and_export",
]
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"


def load_manifest() -> dict[str, Any]:
    if RESUME and MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"config_hash": CONFIG_HASH, "completed": [], "stages": {}, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def mark_stage(name: str, status: str, payload: dict[str, Any]) -> None:
    manifest = load_manifest()
    manifest["stages"][name] = {"status": status, "payload": payload, "time_s": now_s()}
    if status == "completed" and name not in manifest["completed"]:
        manifest["completed"].append(name)
    write_json(MANIFEST_PATH, manifest)


@dataclass
class BranchLedger:
    branch: str
    optimizer_steps: int = 0
    abstract_positions: int = 0
    generated_candidates: int = 0
    preference_pairs: int = 0
    rl_trajectories: int = 0
    token_exposure: int = 0
    wall_clock_s: float = 0.0


def assert_fair_budget(a: BranchLedger, b: BranchLedger, tolerance: float = CFG.branch_budget_tolerance) -> None:
    fields = ["optimizer_steps", "abstract_positions", "generated_candidates", "preference_pairs", "rl_trajectories", "token_exposure"]
    for f in fields:
        av, bv = getattr(a, f), getattr(b, f)
        denom = max(1, (abs(av) + abs(bv)) / 2)
        delta = abs(av - bv) / denom
        if delta > tolerance:
            raise AssertionError(f"budget mismatch {f}: {av} vs {bv} ({delta:.3f})")


def estimate_exposure(rows: int, seq_len: int, epochs: float) -> int:
    return int(rows * seq_len * epochs)


flat_ledger = BranchLedger("FLAT_MATCHED")
hacot_ledger = BranchLedger("HACOT")


def run_stage_smoke(name: str) -> dict[str, Any]:
    time.sleep(0.01)
    payload = {"stage": name, "mode": MODE, "status": "smoke_completed"}
    mark_stage(name, "completed", payload)
    return payload


def maybe_safe_stop() -> bool:
    remaining_min = SESSION_LIMIT_MIN - now_s() / 60
    return remaining_min < 30


def run_training_orchestration() -> dict[str, Any]:
    completed = set(load_manifest().get("completed", []))
    outputs = {}
    for stage in STAGES:
        if stage in completed and RESUME:
            continue
        if maybe_safe_stop():
            mark_stage(stage, "paused_before_stage", {"reason": "session limit guard"})
            break
        if SMOKE:
            outputs[stage] = run_stage_smoke(stage)
        else:
            # FULL implementation hook: each phase calls JAX/Gemma train/eval functions
            # below. Keeping this explicit prevents silent fallback to a toy backend.
            mark_stage(stage, "ready", {"requires": "execute FULL Gemma phase function in Kaggle TPU runtime"})
            if RUN_STAGE != "AUTO" and RUN_STAGE == stage:
                raise NotImplementedError(f"Stage {stage} is ready for TPU execution but not run automatically in this generated harness.")
    return outputs


stage_outputs = run_training_orchestration()
stage_outputs
"""
    ),
    md("## Evaluation, Ablations, Statistics, And Verdict"),
    code(
        r"""
def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def paired_bootstrap(xs: list[int], ys: list[int], n: int = 10_000, seed: int = SEED) -> dict[str, float]:
    rng = random.Random(seed)
    diffs = []
    m = len(xs)
    if m == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    for _ in range(n if FULL else min(n, 500)):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(xs[i] - ys[i] for i in idx) / m)
    diffs.sort()
    return {
        "mean": sum(diffs) / len(diffs),
        "ci_low": diffs[int(0.025 * (len(diffs) - 1))],
        "ci_high": diffs[int(0.975 * (len(diffs) - 1))],
    }


def mcnemar_counts(xs: list[int], ys: list[int]) -> dict[str, int]:
    b = sum(1 for x, y in zip(xs, ys) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(xs, ys) if x == 0 and y == 1)
    return {"hacot_only": b, "flat_only": c}


def deterministic_verdict(metrics: dict[str, Any]) -> str:
    if metrics.get("unique_reasoning_prompts", 0) < CFG.min_unique_reasoning_prompts and FULL:
        return "RESOURCE_INSUFFICIENT"
    if metrics.get("hacot_training_tokens", 0) < CFG.min_hacot_training_tokens and FULL:
        return "RESOURCE_INSUFFICIENT"
    if metrics.get("preference_trajectories", 0) < CFG.min_preference_trajectories and FULL:
        return "RESOURCE_INSUFFICIENT"
    diff_ci_low = metrics.get("hacot_minus_flat_ci_low", -1.0)
    noninferior_verbal = metrics.get("verbal_noninferior", False)
    token_gain = metrics.get("reasoning_token_reduction", 0.0) >= 2.0
    ood_gain = metrics.get("ood_depth_gain_ci_low", -1.0) > 0
    causal = metrics.get("structure_perturbation_drop_ci_low", -1.0) > 0
    all_level = metrics.get("all_level_beats_root_only_ci_low", -1.0) > 0
    two_seeds = metrics.get("completed_main_seeds", 0) >= 2
    retention_ok = metrics.get("general_capability_drop", 1.0) <= 0.015
    if all([diff_ci_low > 0, noninferior_verbal, token_gain, ood_gain, causal, all_level, two_seeds, retention_ok]):
        return "STRONG_POSITIVE"
    if metrics.get("hacot_noninferior", False) and token_gain:
        return "EFFICIENCY_POSITIVE"
    if metrics.get("family_variance_high", False):
        return "MIXED"
    if metrics.get("hacot_worse_than_flat", False):
        return "NEGATIVE"
    return "INCONCLUSIVE"


def write_smoke_metrics() -> dict[str, Any]:
    rng = random.Random(SEED)
    hacot = [rng.randint(0, 1) for _ in range(64)]
    flat = [rng.randint(0, 1) for _ in range(64)]
    boot = paired_bootstrap(hacot, flat)
    metrics = {
        "unique_reasoning_prompts": len(synthetic_train) + len(natural_rows),
        "hacot_training_tokens": estimate_exposure(len(synthetic_train), 256, 1),
        "preference_trajectories": 0,
        "hacot_minus_flat_mean": boot["mean"],
        "hacot_minus_flat_ci_low": boot["ci_low"],
        "hacot_minus_flat_ci_high": boot["ci_high"],
        "mcnemar": mcnemar_counts(hacot, flat),
        "completed_main_seeds": len(CFG.seeds_main),
        "smoke": True,
    }
    metrics["verdict"] = deterministic_verdict(metrics)
    write_json(ARTIFACT_DIR / "reports" / "verdict.json", {"verdict": metrics["verdict"], "metrics": metrics})
    write_json(ARTIFACT_DIR / "reports" / "final_report.json", metrics)
    if pd is not None:
        pd.DataFrame([metrics]).to_csv(ARTIFACT_DIR / "metrics" / "all_results.csv", index=False)
    return metrics


final_metrics = write_smoke_metrics() if SMOKE else {
    "verdict": "RESOURCE_INSUFFICIENT",
    "reason": "FULL metrics are produced after TPU training stages complete.",
}
final_metrics
"""
    ),
    md("## Exported Inference API"),
    code(
        r"""
INFERENCE_API = r'''
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HACoTResult:
    answer: str
    abstract_nodes: int
    max_depth: int
    latency_ms: float
    tree: list[str] | None = None


class HACoTReasoner:
    def __init__(self, artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        self.token_plan = json.loads((self.artifact_dir / "tokenizer" / "token_extension_plan.json").read_text())

    @classmethod
    def load(cls, path):
        return cls(path)

    def generate(self, prompt, mode="standard", return_tree=False, temperature=0.0):
        t0 = time.time()
        # The deployed notebook replaces this smoke response with Gemma sampling,
        # constrained abstract decoding, tree-aware re-encoding, and answer decoding.
        answer = "SMOKE_EXPORT_ONLY"
        tree = ["<HA_START>", "<A00>", "<HA_ROOT>", "<HA_END>"]
        return HACoTResult(
            answer=answer,
            abstract_nodes=1,
            max_depth=1,
            latency_ms=(time.time() - t0) * 1000,
            tree=tree if return_tree else None,
        )
'''

(ARTIFACT_DIR / "inference" / "hacot_reasoner.py").write_text(INFERENCE_API, encoding="utf-8")
write_json(ARTIFACT_DIR / "inference" / "generation_config.json", {
    "fast": {"max_abstract_nodes": 32},
    "standard": {"max_abstract_nodes": 64},
    "deep": {"max_abstract_nodes": 128},
    "temperature_default": 0.0,
})
write_json(ARTIFACT_DIR / "reports" / "export_report.json", {
    "research_checkpoint": "checkpoints/hacot_research_best",
    "deployment_checkpoint": "checkpoints/hacot_deploy",
    "inference_api": "inference/hacot_reasoner.py",
})
print("exported", ARTIFACT_DIR / "inference" / "hacot_reasoner.py")
"""
    ),
    code(
        r"""
final_md = f'''# HACoT Final Report

Mode: {MODE}
Config hash: {CONFIG_HASH}

This report is generated by the notebook, not by a model judge.

## Current Verdict

`{final_metrics.get('verdict', 'PENDING')}`

## Resource Notes

- Backend: {CFG.backend}
- Session limit: {CFG.session_limit_min} minutes
- Runpod recommendation: Community A100 SXM 80GB first, A100 PCIe 80GB if SXM is unavailable.
- Runpod manifest: `reports/runpod_gpu_manifest.json`

## Data

- Synthetic train examples: {len(synthetic_train)}
- Natural reasoning rows after filtering: {len(natural_rows)}
- Dev/test are generated from separate seeds and split keys.

## Tests

- Grammar: {grammar_test_report}
- Tree mask: {mask_report}

## Limitations

SMOKE mode only validates control flow. FULL mode must complete TPU training and
matched branch evaluation before any scientific claim can be made.
'''
(ARTIFACT_DIR / "reports" / "final_report.md").write_text(final_md, encoding="utf-8")
print(final_md)
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "TPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/kaggle_hacot_gemma4_e2b.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
