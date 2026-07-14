"""Build a Kaggle notebook for a stricter native recursive thought-token benchmark."""

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
# Native Recursive Thought Tokens: Programmed Graph Benchmark

This notebook is a stricter follow-up to the previous recursive thought-token notebooks.

The goal is not to retrofit a frozen LLM. The goal is to test the core architectural claim in a
controlled native setting:

> A reusable latent thought state, applied recursively, should learn compositional reasoning more
> efficiently and extrapolate better than a flat transformer with a comparable parameter budget.

What is improved compared with the earlier notebooks:

- the branch task is no longer ambiguous: traps are distinguishable by relation or current state;
- every problem contains a visible relation program, so variable depth is part of the input;
- recursive models are evaluated three ways: final fixed-step output, oracle-depth output, and
  learned halting output;
- the notebook reports step-wise attention accuracy, node-state accuracy, halting accuracy, depth
  breakdowns, throughput, parameter efficiency, and multi-seed means;
- baselines include a full flat transformer, a small flat transformer, recursive with no auxiliary
  supervision, recursive with thought-level supervision, and recursive with annealed supervision.

Set `FAST_RUN=1` for a quick syntax/debug run. The default profile is intended for a Kaggle T4.
"""
    ),
    code(
        r"""
import math, os, random, time, json, statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FAST_RUN = os.environ.get('FAST_RUN', '0') == '1'
PROFILE = os.environ.get('RTT_PROFILE', 'NATIVE_RECURSIVE_BENCHMARK_T4')

if FAST_RUN:
    CFG = dict(
        train_steps=80,
        batch_size=96,
        eval_batches=3,
        seeds=[1],
        d_model=96,
        n_entities=64,
        n_relations=5,
        max_steps=8,
        distractors=10,
    )
else:
    CFG = dict(
        train_steps=1800,
        batch_size=256,
        eval_batches=16,
        seeds=[1, 2, 3],
        d_model=128,
        n_entities=96,
        n_relations=6,
        max_steps=10,
        distractors=18,
    )

LR = 3e-4
TRAIN_DEPTH = (1, 4)
OOD_DEPTH = (5, CFG['max_steps'])
HARD_DEPTH = (CFG['max_steps'] + 1, CFG['max_steps'] + 4)
TASKS = ['chain', 'program', 'branch']
STOP_REL = CFG['n_relations']
PAD_REL = CFG['n_relations'] + 1
REL_VOCAB = CFG['n_relations'] + 2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
print('profile:', PROFILE, 'FAST_RUN:', FAST_RUN)
print(json.dumps(CFG, indent=2))
"""
    ),
    md(
        r"""
## Data

Each example is a small graph plus a program.

Input:

- start entity `q`;
- relation program `r_1, r_2, ..., r_depth`;
- shuffled facts `(source, relation, target)`.

Target:

- entity reached after applying the program from the start entity.

Tasks:

- `chain`: all gold relations are `0`, a simple repeated traversal;
- `program`: each step can use a different relation;
- `branch`: adds tempting traps from current path nodes, but unlike earlier notebooks these traps
  remain logically distinguishable. There is at most one fact for a given `(source, relation)` key.
"""
    ),
    code(
        r"""
@dataclass
class Batch:
    facts: torch.Tensor       # [B, F, 3] = source, relation, target
    program: torch.Tensor     # [B, S] relation ids, STOP after depth
    query: torch.Tensor       # [B]
    answer: torch.Tensor      # [B]
    depth: torch.Tensor       # [B]
    path_pos: torch.Tensor    # [B, S], -100 after depth
    path_nodes: torch.Tensor  # [B, S + 1], -100 after depth + 1
    task_id: torch.Tensor     # [B]


TASK_TO_ID = {name: i for i, name in enumerate(TASKS)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rand_entity_not(value: int, n_entities: int) -> int:
    out = random.randrange(n_entities)
    while out == value:
        out = random.randrange(n_entities)
    return out


def make_batch(batch_size: int, min_depth: int, max_depth: int, *, task='mixed', device=device) -> Batch:
    n_entities = CFG['n_entities']
    n_relations = CFG['n_relations']
    max_steps = max(CFG['max_steps'], max_depth)
    n_facts = max_depth + CFG['distractors'] + max_depth

    facts_list, programs, queries, answers, depths, pos_list, node_list, task_ids = [], [], [], [], [], [], [], []

    for _ in range(batch_size):
        task_name = random.choice(TASKS) if task == 'mixed' else task
        depth = random.randint(min_depth, max_depth)
        start = random.randrange(n_entities)
        nodes = [start]
        used_nodes = {start}
        for _step in range(depth):
            nxt = random.randrange(n_entities)
            tries = 0
            while nxt in used_nodes and tries < 50:
                nxt = random.randrange(n_entities)
                tries += 1
            nodes.append(nxt)
            used_nodes.add(nxt)

        if task_name == 'chain':
            rels = [0 for _ in range(depth)]
        else:
            rels = [random.randrange(n_relations) for _ in range(depth)]

        facts = []
        keys = set()
        gold_indices = []

        def add_fact(src, rel, tgt):
            key = (src, rel)
            if key in keys:
                return False
            keys.add(key)
            facts.append([src, rel, tgt])
            return True

        for step, rel in enumerate(rels):
            gold_indices.append(len(facts))
            ok = add_fact(nodes[step], rel, nodes[step + 1])
            assert ok

        if task_name == 'branch':
            for step, rel in enumerate(rels):
                # Trap 1: same source, wrong relation, wrong target. Looks local but program rejects it.
                wrong_rel = random.randrange(n_relations)
                tries = 0
                while (wrong_rel == rel or (nodes[step], wrong_rel) in keys) and tries < 50:
                    wrong_rel = random.randrange(n_relations)
                    tries += 1
                if wrong_rel != rel:
                    add_fact(nodes[step], wrong_rel, _rand_entity_not(nodes[step + 1], n_entities))

                # Trap 2: correct relation, nearby/wrong source. Looks relation-compatible but state rejects it.
                wrong_src = _rand_entity_not(nodes[step], n_entities)
                tries = 0
                while (wrong_src, rel) in keys and tries < 50:
                    wrong_src = _rand_entity_not(nodes[step], n_entities)
                    tries += 1
                add_fact(wrong_src, rel, random.randrange(n_entities))

        attempts = 0
        while len(facts) < n_facts and attempts < n_facts * 50:
            attempts += 1
            src = random.randrange(n_entities)
            rel = random.randrange(n_relations)
            tgt = random.randrange(n_entities)
            add_fact(src, rel, tgt)

        while len(facts) < n_facts:
            # Rare fallback if the key space is almost full.
            facts.append([random.randrange(n_entities), random.randrange(n_relations), random.randrange(n_entities)])

        perm = list(range(len(facts)))
        random.shuffle(perm)
        shuffled = [facts[i] for i in perm]
        old_to_new = {old: new for new, old in enumerate(perm)}

        program = rels + [STOP_REL] * max(0, max_steps - depth)
        path_pos = [old_to_new[i] for i in gold_indices] + [-100] * max(0, max_steps - depth)
        path_nodes = nodes + [-100] * max(0, max_steps + 1 - len(nodes))

        facts_list.append(shuffled)
        programs.append(program[:max_steps])
        queries.append(start)
        answers.append(nodes[-1])
        depths.append(depth)
        pos_list.append(path_pos[:max_steps])
        node_list.append(path_nodes[:max_steps + 1])
        task_ids.append(TASK_TO_ID[task_name])

    return Batch(
        facts=torch.tensor(facts_list, dtype=torch.long, device=device),
        program=torch.tensor(programs, dtype=torch.long, device=device),
        query=torch.tensor(queries, dtype=torch.long, device=device),
        answer=torch.tensor(answers, dtype=torch.long, device=device),
        depth=torch.tensor(depths, dtype=torch.long, device=device),
        path_pos=torch.tensor(pos_list, dtype=torch.long, device=device),
        path_nodes=torch.tensor(node_list, dtype=torch.long, device=device),
        task_id=torch.tensor(task_ids, dtype=torch.long, device=device),
    )


def show_example(task='branch'):
    b = make_batch(1, 3, 3, task=task)
    print('task:', TASKS[int(b.task_id[0])], 'query:', int(b.query[0]), 'answer:', int(b.answer[0]), 'depth:', int(b.depth[0]))
    print('program:', b.program[0].detach().cpu().tolist())
    print('facts:', b.facts[0].detach().cpu().tolist())
    print('gold fact positions:', b.path_pos[0].detach().cpu().tolist())
    print('gold path nodes:', b.path_nodes[0].detach().cpu().tolist())


show_example('branch')
"""
    ),
    md(
        r"""
## Models

The flat transformer receives the whole graph and program as ordinary tokens.

The recursive thought model keeps one latent state. At every step it combines:

- current thought state;
- current program relation;
- attention over graph facts.

The state after each step is an intermediate thought token. The model is trained/evaluated with
three answer modes:

- `final`: output after all fixed steps;
- `oracle_depth`: output at the true program depth;
- `halt`: output selected by a learned halting distribution.
"""
    ),
    code(
        r"""
class FlatProgramTransformer(nn.Module):
    def __init__(self, d_model=CFG['d_model'], n_layers=4, n_heads=4, max_facts=96, max_steps=16):
        super().__init__()
        self.entity = nn.Embedding(CFG['n_entities'], d_model)
        self.relation = nn.Embedding(REL_VOCAB, d_model)
        self.type_embed = nn.Embedding(5, d_model)
        self.pos = nn.Parameter(torch.randn(1, 2 + max_steps + max_facts, d_model) * 0.02)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.05,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, CFG['n_entities'])

    def encode_edges(self, facts):
        return self.edge_mlp(torch.cat([
            self.entity(facts[:, :, 0]),
            self.relation(facts[:, :, 1]),
            self.entity(facts[:, :, 2]),
        ], dim=-1))

    def forward(self, batch: Batch):
        B, F_, _ = batch.facts.shape
        S = batch.program.shape[1]
        cls = torch.zeros(B, 1, self.pos.shape[-1], device=batch.facts.device)
        query = self.entity(batch.query)[:, None, :]
        program = self.relation(batch.program)
        edges = self.encode_edges(batch.facts)
        x = torch.cat([cls, query, program, edges], dim=1)
        types = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=batch.facts.device),
            torch.ones(B, 1, dtype=torch.long, device=batch.facts.device),
            torch.full((B, S), 2, dtype=torch.long, device=batch.facts.device),
            torch.full((B, F_), 3, dtype=torch.long, device=batch.facts.device),
        ], dim=1)
        x = x + self.type_embed(types) + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(x)[:, 0])


class RecursiveProgramThoughtModel(nn.Module):
    def __init__(self, d_model=CFG['d_model'], max_steps=CFG['max_steps']):
        super().__init__()
        self.max_steps = max_steps
        self.entity = nn.Embedding(CFG['n_entities'], d_model)
        self.relation = nn.Embedding(REL_VOCAB, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.key_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.query_mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.value_mlp = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.update = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.node_head = nn.Linear(d_model, CFG['n_entities'])
        self.halt_head = nn.Linear(d_model, 1)

    def encode_facts(self, facts):
        src = self.entity(facts[:, :, 0])
        rel = self.relation(facts[:, :, 1])
        tgt = self.entity(facts[:, :, 2])
        edge = self.edge_mlp(torch.cat([src, rel, tgt], dim=-1))
        key = self.key_mlp(torch.cat([src, rel], dim=-1))
        value = self.value_mlp(torch.cat([edge, tgt, rel], dim=-1))
        return key, value

    def forward(self, batch: Batch, return_trace=False):
        B = batch.query.shape[0]
        steps = batch.program.shape[1]
        keys, values = self.encode_facts(batch.facts)
        state = self.entity(batch.query)
        attns, states, node_logits, halt_logits = [], [state], [], []
        scale = math.sqrt(state.shape[-1])

        for t in range(steps):
            rel_emb = self.relation(batch.program[:, t])
            query = self.query_mlp(torch.cat([state, rel_emb], dim=-1))[:, None, :]
            score = (query * keys).sum(-1) / scale
            attn = torch.softmax(score, dim=-1)
            ctx = torch.einsum('bf,bfd->bd', attn, values)
            stop = (batch.program[:, t] == STOP_REL).float()[:, None]
            proposed = self.norm(self.update(ctx, state))
            state = stop * state + (1.0 - stop) * proposed
            attns.append(attn)
            states.append(state)
            node_logits.append(self.node_head(state))
            halt_logits.append(self.halt_head(state).squeeze(-1))

        node_logits_t = torch.stack(node_logits, dim=1)
        halt_logits_t = torch.stack(halt_logits, dim=1)
        halt_weights = torch.softmax(halt_logits_t, dim=1)
        halt_logits_weighted = torch.einsum('bs,bse->be', halt_weights, node_logits_t)
        final_logits = node_logits_t[:, -1]
        oracle_idx = (batch.depth - 1).clamp_min(0)
        oracle_logits = node_logits_t[torch.arange(B, device=batch.facts.device), oracle_idx]

        if not return_trace:
            return halt_logits_weighted
        return halt_logits_weighted, {
            'final_logits': final_logits,
            'oracle_logits': oracle_logits,
            'node_logits': node_logits_t,
            'halt_logits': halt_logits_t,
            'halt_weights': halt_weights,
            'attn': torch.stack(attns, dim=1),
            'states': torch.stack(states, dim=1),
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for name, model in [
    ('flat_full', FlatProgramTransformer()),
    ('flat_small', FlatProgramTransformer(d_model=max(64, CFG['d_model'] // 2), n_layers=3)),
    ('recursive', RecursiveProgramThoughtModel()),
]:
    print(name, count_params(model))
"""
    ),
    md("## Losses and Metrics"),
    code(
        r"""
def recursive_loss(model, batch: Batch, aux_mode: str, step: int, total_steps: int):
    logits, trace = model(batch, return_trace=True)
    answer_loss = F.cross_entropy(logits, batch.answer)
    loss = answer_loss
    parts = {'answer_loss': float(answer_loss.detach().cpu())}

    if aux_mode != 'none':
        valid = batch.path_pos >= 0
        if valid.any():
            flat_valid = valid.reshape(-1)
            attn_logits = torch.log(trace['attn'].clamp_min(1e-8)).reshape(-1, batch.facts.shape[1])
            node_logits = trace['node_logits'].reshape(-1, CFG['n_entities'])
            attn_targets = batch.path_pos.reshape(-1)
            node_targets = batch.path_nodes[:, 1:].reshape(-1)
            attn_loss = F.nll_loss(attn_logits[flat_valid], attn_targets[flat_valid])
            node_loss = F.cross_entropy(node_logits[flat_valid], node_targets[flat_valid])
        else:
            attn_loss = answer_loss.new_tensor(0.0)
            node_loss = answer_loss.new_tensor(0.0)
        halt_target = (batch.depth - 1).clamp(0, trace['halt_logits'].shape[1] - 1)
        halt_loss = F.cross_entropy(trace['halt_logits'], halt_target)

        if aux_mode == 'annealed':
            frac = step / max(1, total_steps - 1)
            aux_weight = max(0.1, 1.0 - frac)
        else:
            aux_weight = 1.0

        loss = loss + aux_weight * (0.4 * attn_loss + 0.4 * node_loss + 0.2 * halt_loss)
        parts.update({
            'aux_weight': float(aux_weight),
            'attn_loss': float(attn_loss.detach().cpu()),
            'node_loss': float(node_loss.detach().cpu()),
            'halt_loss': float(halt_loss.detach().cpu()),
        })

    return loss, parts


@torch.no_grad()
def evaluate(model, model_type: str, min_depth: int, max_depth: int, *, task='mixed', batches=CFG['eval_batches']):
    model.eval()
    total = 0
    correct = {'answer_acc': 0, 'final_acc': 0, 'oracle_depth_acc': 0, 'halt_argmax_acc': 0}
    path_correct = 0
    path_total = 0
    node_correct = 0
    node_total = 0
    halt_correct = 0
    depth_stats = {}
    start = time.time()

    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_depth, max_depth, task=task)
        B = batch.query.shape[0]
        total += B
        if model_type == 'flat':
            logits = model(batch)
            pred = logits.argmax(-1)
            hit = pred.eq(batch.answer)
            correct['answer_acc'] += int(hit.sum().item())
            correct['final_acc'] += int(hit.sum().item())
            correct['oracle_depth_acc'] += int(hit.sum().item())
            correct['halt_argmax_acc'] += int(hit.sum().item())
        else:
            logits, trace = model(batch, return_trace=True)
            pred = logits.argmax(-1)
            final_pred = trace['final_logits'].argmax(-1)
            oracle_pred = trace['oracle_logits'].argmax(-1)
            halt_idx = trace['halt_logits'].argmax(-1)
            halt_pred = trace['node_logits'][torch.arange(B, device=device), halt_idx].argmax(-1)
            correct['answer_acc'] += int(pred.eq(batch.answer).sum().item())
            correct['final_acc'] += int(final_pred.eq(batch.answer).sum().item())
            correct['oracle_depth_acc'] += int(oracle_pred.eq(batch.answer).sum().item())
            correct['halt_argmax_acc'] += int(halt_pred.eq(batch.answer).sum().item())
            halt_correct += int(halt_idx.eq(batch.depth - 1).sum().item())

            valid = batch.path_pos >= 0
            if valid.any():
                attn_pred = trace['attn'].argmax(-1)
                path_correct += int(attn_pred[valid].eq(batch.path_pos[valid]).sum().item())
                path_total += int(valid.sum().item())
                node_pred = trace['node_logits'].argmax(-1)
                node_target = batch.path_nodes[:, 1:]
                node_correct += int(node_pred[valid].eq(node_target[valid]).sum().item())
                node_total += int(valid.sum().item())

        for d in batch.depth.detach().cpu().tolist():
            depth_stats.setdefault(int(d), {'n': 0, 'answer': 0})
        hits = pred.eq(batch.answer).detach().cpu().tolist()
        for d, h in zip(batch.depth.detach().cpu().tolist(), hits):
            depth_stats[int(d)]['n'] += 1
            depth_stats[int(d)]['answer'] += int(h)

    seconds = time.time() - start
    out = {k: v / total for k, v in correct.items()}
    out['halt_acc'] = halt_correct / total if model_type != 'flat' else None
    out['path_attn_acc'] = path_correct / path_total if path_total else None
    out['node_state_acc'] = node_correct / node_total if node_total else None
    out['examples_per_s'] = total / max(seconds, 1e-9)
    out['by_depth'] = {str(d): depth_stats[d]['answer'] / depth_stats[d]['n'] for d in sorted(depth_stats)}
    return out


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    return {'mean': float(statistics.mean(values)), 'std': float(statistics.pstdev(values)), 'n': len(values)}
"""
    ),
    md("## Training Loop"),
    code(
        r"""
def make_model(spec):
    if spec == 'flat_full':
        return FlatProgramTransformer(), 'flat', 'none'
    if spec == 'flat_small':
        return FlatProgramTransformer(d_model=max(64, CFG['d_model'] // 2), n_layers=3), 'flat', 'none'
    if spec == 'recursive_e2e':
        return RecursiveProgramThoughtModel(), 'recursive', 'none'
    if spec == 'recursive_aux':
        return RecursiveProgramThoughtModel(), 'recursive', 'aux'
    if spec == 'recursive_annealed':
        return RecursiveProgramThoughtModel(), 'recursive', 'annealed'
    raise ValueError(spec)


def train_one(spec: str, seed: int):
    set_seed(seed)
    model, model_type, aux_mode = make_model(spec)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    history = []
    t0 = time.time()

    for step in range(CFG['train_steps']):
        model.train()
        batch = make_batch(CFG['batch_size'], *TRAIN_DEPTH, task='mixed')
        opt.zero_grad(set_to_none=True)
        if model_type == 'flat':
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.answer)
            parts = {'answer_loss': float(loss.detach().cpu())}
        else:
            loss, parts = recursive_loss(model, batch, aux_mode, step, CFG['train_steps'])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 0 or (step + 1) % max(1, CFG['train_steps'] // 6) == 0:
            with torch.no_grad():
                quick = evaluate(model, model_type, *TRAIN_DEPTH, task='mixed', batches=1)
            row = {
                'step': step + 1,
                'loss': float(loss.detach().cpu()),
                'grad_norm': float(torch.as_tensor(grad_norm).detach().cpu()),
                'elapsed_s': time.time() - t0,
                **parts,
                'quick_train_acc': quick['answer_acc'],
            }
            history.append(row)
            print(spec, 'seed', seed, json.dumps(row))

    evals = {}
    for split, depths in [('iid', TRAIN_DEPTH), ('ood', OOD_DEPTH), ('hard', HARD_DEPTH)]:
        evals[split] = evaluate(model, model_type, *depths, task='mixed')
    for task in TASKS:
        evals[f'ood_{task}'] = evaluate(model, model_type, *OOD_DEPTH, task=task)

    return {
        'spec': spec,
        'seed': seed,
        'model_type': model_type,
        'aux_mode': aux_mode,
        'params': count_params(model),
        'history': history,
        'evals': evals,
    }


SPECS = ['flat_full', 'flat_small', 'recursive_e2e', 'recursive_aux', 'recursive_annealed']
all_runs = []
suite_start = time.time()
for spec in SPECS:
    for seed in CFG['seeds']:
        run = train_one(spec, seed)
        all_runs.append(run)
        print('DONE', spec, 'seed', seed, 'ood', run['evals']['ood']['answer_acc'], 'hard', run['evals']['hard']['answer_acc'])

print('suite seconds:', time.time() - suite_start)
"""
    ),
    md("## Summary"),
    code(
        r"""
def summarize_runs(runs):
    summary = {}
    for spec in sorted({r['spec'] for r in runs}):
        rows = [r for r in runs if r['spec'] == spec]
        item = {
            'params': mean_std([r['params'] for r in rows]),
        }
        for split in ['iid', 'ood', 'hard', 'ood_chain', 'ood_program', 'ood_branch']:
            for metric in ['answer_acc', 'final_acc', 'oracle_depth_acc', 'halt_argmax_acc', 'halt_acc', 'path_attn_acc', 'node_state_acc', 'examples_per_s']:
                item[f'{split}_{metric}'] = mean_std([r['evals'][split].get(metric) for r in rows])
        params_mean = item['params']['mean']
        ood_mean = item['ood_answer_acc']['mean']
        hard_mean = item['hard_answer_acc']['mean']
        item['ood_acc_per_million_params'] = None if not params_mean or ood_mean is None else ood_mean / (params_mean / 1_000_000)
        item['hard_acc_per_million_params'] = None if not params_mean or hard_mean is None else hard_mean / (params_mean / 1_000_000)
        summary[spec] = item
    return summary


summary = summarize_runs(all_runs)
print(json.dumps(summary, indent=2))

report = {
    'version': 'native_recursive_program_benchmark_v1',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'profile': PROFILE,
    'fast_run': FAST_RUN,
    'device': str(device),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'config': CFG,
    'train_depth': TRAIN_DEPTH,
    'ood_depth': OOD_DEPTH,
    'hard_depth': HARD_DEPTH,
    'tasks': TASKS,
    'runs': all_runs,
    'summary': summary,
}

out_path = Path('/kaggle/working/native_recursive_program_benchmark_report.json')
if not Path('/kaggle/working').exists():
    out_path = Path('native_recursive_program_benchmark_report.json')
out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('wrote', out_path)
"""
    ),
    md("## Plots"),
    code(
        r"""
labels = list(summary.keys())
ood = [summary[k]['ood_answer_acc']['mean'] for k in labels]
hard = [summary[k]['hard_answer_acc']['mean'] for k in labels]
eff = [summary[k]['ood_acc_per_million_params'] for k in labels]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(np.arange(len(labels)) - 0.18, ood, width=0.36, label='OOD')
axes[0].bar(np.arange(len(labels)) + 0.18, hard, width=0.36, label='Hard')
axes[0].set_xticks(np.arange(len(labels)))
axes[0].set_xticklabels(labels, rotation=35, ha='right')
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel('accuracy')
axes[0].legend()
axes[0].set_title('Depth generalization')

axes[1].bar(labels, eff)
axes[1].set_xticklabels(labels, rotation=35, ha='right')
axes[1].set_ylabel('OOD accuracy / M params')
axes[1].set_title('Parameter efficiency')
plt.tight_layout()
plt.show()

for spec in labels:
    print('\n', spec)
    print('OOD mixed by depth:', all_runs[[r['spec'] for r in all_runs].index(spec)]['evals']['ood']['by_depth'])
"""
    ),
    md(
        r"""
## Interpretation Guide

The result supports the recursive thought-token hypothesis only if:

- `recursive_aux` or `recursive_annealed` beats `flat_small` and is competitive with or better than
  `flat_full` on OOD/hard depth;
- the gain survives the `branch` split, not only `chain`;
- `oracle_depth_acc` is high but `halt_argmax_acc` is low only when halting is the bottleneck;
- `path_attn_acc` and intermediate node accuracy rise together with answer accuracy;
- the advantage is visible across seeds, not one lucky run.

Negative outcomes are also informative:

- high IID but low OOD means the model memorized shallow programs;
- high oracle-depth but low final/halt means composition works but stopping is weak;
- high path attention but low answer means state update is weak;
- flat baselines winning at matched or lower parameter count means recursion is not yet buying
  useful inductive bias for this task family.
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("notebooks/kaggle_native_recursive_benchmark.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
