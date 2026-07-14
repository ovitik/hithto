"""Build the Kaggle proof notebook for recursive thought-token experiments."""

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
# Recursive Thought Tokens: Ultimate Kaggle GPU Stress Test v3

This notebook is a stronger proof-of-principle for the original hypothesis:

> Higher-level recursive thought tokens can represent composed thoughts more efficiently than ordinary flat tokens, improving reasoning quality and/or speed.

It deliberately tests both supportive and falsifying cases:

- **Parameter efficiency**: does a recursive thought model beat a parameter-matched flat Transformer?
- **Depth extrapolation**: does it generalize from short chains to deeper reasoning chains?
- **Training signal bottleneck**: is failure caused by architecture, capacity, or weak credit assignment?
- **Stopping confound**: does the model fail because it cannot compose thoughts, or because it does not know when to stop?
- **Oracle dependence**: does explicit step supervision help, can it be annealed away, and can learned halting replace it?
- **Hierarchy/recursion**: do intermediate states become useful thoughts rather than unused hidden activations?
- **Harder structure**: does the result survive distractors, branching traps, multiple relation types, and longer depth?
- **Robustness**: does the conclusion hold over multiple seeds?
- **Efficiency**: what accuracy do we get per parameter and per millisecond?

Default profile is intended for Kaggle GPU and should run materially longer than the earlier 2-minute notebook. Set `FAST_RUN=True` for a quick syntax/debug run.
"""
    ),
    code(
        r"""
import math, os, random, time, json, statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kaggle-friendly controls.
FAST_RUN = os.environ.get('FAST_RUN', '0') == '1'
PROFILE = os.environ.get('RTT_PROFILE', 'ULTIMATE_20MIN')

if FAST_RUN:
    CFG = dict(
        train_steps=80,
        aux_steps=100,
        sweep_steps=60,
        batch_size=96,
        eval_batches=3,
        seeds=[1],
        d_model=96,
        n_entities=64,
        max_steps=8,
        distractors=8,
    )
else:
    # Designed for a single Kaggle T4/P100 class GPU. Increase steps/seeds if you want a longer run.
    CFG = dict(
        train_steps=1600,
        aux_steps=1800,
        sweep_steps=900,
        batch_size=256,
        eval_batches=16,
        seeds=[1, 2, 3],
        d_model=128,
        n_entities=96,
        max_steps=10,
        distractors=16,
    )

LR = 3e-4
N_RELATIONS = 4
TRAIN_DEPTH = (1, 3)
OOD_DEPTH = (4, CFG['max_steps'])
HARD_DEPTH = (CFG['max_steps'] + 1, CFG['max_steps'] + 4)

print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
print('profile:', PROFILE, 'FAST_RUN:', FAST_RUN)
print(json.dumps(CFG, indent=2))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
"""
    ),
    md("## Dataset\n\nThe generator creates shuffled symbolic reasoning problems. A query asks: starting from entity `q`, follow a hidden chain of length `depth`; predict the final entity. Facts are unordered directed edges with relation labels. Distractors and trap edges force the model to retrieve and compose the right fact at each recursive step."),
    code(
        r"""
@dataclass
class Batch:
    facts: torch.Tensor      # [B, F, 3] = source, relation, target
    query: torch.Tensor      # [B]
    answer: torch.Tensor     # [B]
    depth: torch.Tensor      # [B]
    path_pos: torch.Tensor   # [B, max_steps], -100 where padded
    path_nodes: torch.Tensor # [B, max_steps + 1], query and next entities, -100 where padded


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(
    batch_size: int,
    min_depth: int,
    max_depth: int,
    *,
    n_entities: int = CFG['n_entities'],
    n_relations: int = N_RELATIONS,
    distractors: int = CFG['distractors'],
    max_steps: int = CFG['max_steps'],
    task: str = 'chain',
    device=device,
) -> Batch:
    # task='chain': all gold edges have relation 0.
    # task='chain_selfloop': after the gold chain, a terminal answer->answer edge lets fixed-step models stop implicitly.
    # task='multirelation': gold relation changes by step; query must be carried in latent state.
    # task='branch_trap': adds wrong outgoing edges from path nodes with gold relation, so entity state matters.
    base_task = task[:-9] if task.endswith('_selfloop') else task
    use_selfloop = task.endswith('_selfloop')
    effective_steps = max(max_steps, max_depth)
    facts_list, queries, answers, depths, pos_list, node_list = [], [], [], [], [], []
    n_facts = max_depth + distractors + (max_depth if base_task == 'branch_trap' else 0) + (1 if use_selfloop else 0)

    for _ in range(batch_size):
        depth = random.randint(min_depth, max_depth)
        start = random.randrange(n_entities)
        nodes = [start]
        used = {start}
        for _step in range(depth):
            nxt = random.randrange(n_entities)
            tries = 0
            while nxt in used and tries < 50:
                nxt = random.randrange(n_entities)
                tries += 1
            nodes.append(nxt)
            used.add(nxt)

        facts = []
        gold_indices_before_shuffle = []
        for step in range(depth):
            rel = 0 if base_task in ('chain', 'branch_trap') else (step % n_relations)
            gold_indices_before_shuffle.append(len(facts))
            facts.append([nodes[step], rel, nodes[step + 1]])

            if base_task == 'branch_trap':
                wrong = random.randrange(n_entities)
                while wrong == nodes[step + 1]:
                    wrong = random.randrange(n_entities)
                # Same source/relation, wrong target: attention must identify the correct edge among traps.
                facts.append([nodes[step], rel, wrong])

        if use_selfloop:
            self_loop_index = len(facts)
            facts.append([nodes[-1], 0, nodes[-1]])
        else:
            self_loop_index = None

        while len(facts) < n_facts:
            s = random.randrange(n_entities)
            r = random.randrange(n_relations)
            t = random.randrange(n_entities)
            if [s, r, t] not in facts:
                facts.append([s, r, t])

        perm = list(range(len(facts)))
        random.shuffle(perm)
        shuffled = [facts[i] for i in perm]
        old_to_new = {old: new for new, old in enumerate(perm)}
        if use_selfloop:
            path_pos = [old_to_new[i] for i in gold_indices_before_shuffle]
            path_pos += [old_to_new[self_loop_index]] * max(0, effective_steps - depth)
            path_nodes = nodes + [nodes[-1]] * max(0, effective_steps + 1 - len(nodes))
        else:
            path_pos = [old_to_new[i] for i in gold_indices_before_shuffle] + [-100] * max(0, effective_steps - depth)
            path_nodes = nodes + [-100] * max(0, effective_steps + 1 - len(nodes))

        facts_list.append(shuffled)
        queries.append(start)
        answers.append(nodes[-1])
        depths.append(depth)
        pos_list.append(path_pos[:effective_steps])
        node_list.append(path_nodes[:effective_steps + 1])

    return Batch(
        facts=torch.tensor(facts_list, dtype=torch.long, device=device),
        query=torch.tensor(queries, dtype=torch.long, device=device),
        answer=torch.tensor(answers, dtype=torch.long, device=device),
        depth=torch.tensor(depths, dtype=torch.long, device=device),
        path_pos=torch.tensor(pos_list, dtype=torch.long, device=device),
        path_nodes=torch.tensor(node_list, dtype=torch.long, device=device),
    )


def describe_batch(batch: Batch, i=0):
    print('query:', int(batch.query[i]), 'answer:', int(batch.answer[i]), 'depth:', int(batch.depth[i]))
    print('facts:', batch.facts[i].detach().cpu().tolist())
    print('gold fact positions:', batch.path_pos[i].detach().cpu().tolist())
    print('gold path nodes:', batch.path_nodes[i].detach().cpu().tolist())


describe_batch(make_batch(1, 2, 3, task='chain_selfloop'))
"""
    ),
    md("## Models\n\nThe flat baseline sees all facts as ordinary tokens. The recursive model keeps one latent thought state and repeatedly updates it by attending over facts. This is the native version of the original idea: the recurring state is a higher-level token that composes lower-level facts into a deeper thought."),
    code(
        r"""
class BaselineTransformer(nn.Module):
    def __init__(
        self,
        n_entities=CFG['n_entities'],
        n_relations=N_RELATIONS,
        d_model=CFG['d_model'],
        n_heads=4,
        n_layers=3,
        dropout=0.05,
        max_facts=64,
    ):
        super().__init__()
        self.n_entities = n_entities
        self.entity = nn.Embedding(n_entities, d_model)
        self.relation = nn.Embedding(n_relations, d_model)
        self.type_embed = nn.Embedding(5, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1 + max_facts * 3, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_entities)

    def forward(self, facts, query):
        B, F_, _ = facts.shape
        src, rel, tgt = facts[:, :, 0], facts[:, :, 1], facts[:, :, 2]
        tokens = torch.cat([query[:, None], src, rel, tgt], dim=1)
        types = torch.cat([
            torch.zeros((B, 1), dtype=torch.long, device=facts.device),
            torch.ones((B, F_), dtype=torch.long, device=facts.device),
            torch.full((B, F_), 2, dtype=torch.long, device=facts.device),
            torch.full((B, F_), 3, dtype=torch.long, device=facts.device),
        ], dim=1)
        x = torch.empty(B, tokens.shape[1], self.entity.embedding_dim, device=facts.device)
        x[:, 0] = self.entity(tokens[:, 0])
        x[:, 1:1 + F_] = self.entity(src)
        x[:, 1 + F_:1 + 2 * F_] = self.relation(rel)
        x[:, 1 + 2 * F_:] = self.entity(tgt)
        x = x + self.type_embed(types) + self.pos[:, :tokens.shape[1]]
        x = self.encoder(x)
        return self.head(x[:, 0])


class EdgeTransformerBaseline(nn.Module):
    # Fairer flat baseline: each fact is represented as one edge token, preserving triples.
    def __init__(
        self,
        n_entities=CFG['n_entities'],
        n_relations=N_RELATIONS,
        d_model=CFG['d_model'],
        n_heads=4,
        n_layers=3,
        dropout=0.05,
        max_facts=64,
        use_depth=False,
        max_depth=32,
    ):
        super().__init__()
        self.uses_depth = use_depth
        self.entity = nn.Embedding(n_entities, d_model)
        self.relation = nn.Embedding(n_relations, d_model)
        self.depth_embed = nn.Embedding(max_depth + 1, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.type_embed = nn.Embedding(3, d_model)
        self.pos = nn.Parameter(torch.randn(1, 2 + max_facts, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_entities)

    def forward(self, facts, query, depth=None):
        B, F_, _ = facts.shape
        src = self.entity(facts[:, :, 0])
        rel = self.relation(facts[:, :, 1])
        tgt = self.entity(facts[:, :, 2])
        edge = self.edge_mlp(torch.cat([src, rel, tgt], dim=-1))
        q = self.entity(query)[:, None, :]
        if self.uses_depth:
            if depth is None:
                raise ValueError('DepthAware edge baseline requires batch.depth')
            d = self.depth_embed(depth.clamp(0, self.depth_embed.num_embeddings - 1))[:, None, :]
            x = torch.cat([q, d, edge], dim=1)
            types = torch.cat([
                torch.zeros((B, 1), dtype=torch.long, device=facts.device),
                torch.full((B, 1), 2, dtype=torch.long, device=facts.device),
                torch.ones((B, F_), dtype=torch.long, device=facts.device),
            ], dim=1)
        else:
            x = torch.cat([q, edge], dim=1)
            types = torch.cat([
                torch.zeros((B, 1), dtype=torch.long, device=facts.device),
                torch.ones((B, F_), dtype=torch.long, device=facts.device),
            ], dim=1)
        x = x + self.type_embed(types) + self.pos[:, :x.shape[1]]
        x = self.encoder(x)
        return self.head(x[:, 0])


class RecursiveThoughtModel(nn.Module):
    def __init__(
        self,
        n_entities=CFG['n_entities'],
        n_relations=N_RELATIONS,
        d_model=CFG['d_model'],
        max_steps=CFG['max_steps'],
        n_heads=4,
        dropout=0.05,
    ):
        super().__init__()
        self.n_entities = n_entities
        self.max_steps = max_steps
        self.entity = nn.Embedding(n_entities, d_model)
        self.relation = nn.Embedding(n_relations, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.update = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_entities, bias=False)
        self.node_head = nn.Linear(d_model, n_entities)
        self.halt_head = nn.Linear(d_model, 1)
        self.out.weight = self.entity.weight

    def encode_edges(self, facts):
        src = self.entity(facts[:, :, 0])
        rel = self.relation(facts[:, :, 1])
        tgt = self.entity(facts[:, :, 2])
        return self.edge_mlp(torch.cat([src, rel, tgt], dim=-1))

    def forward(self, facts, query, steps=None, return_trace=False):
        steps = self.max_steps if steps is None else steps
        edges = self.encode_edges(facts)
        keys = self.key_proj(edges)
        vals = self.value_proj(edges)
        state = self.entity(query)
        attns, states, node_logits, halt_logits = [], [state], [], []
        scale = math.sqrt(state.shape[-1])
        for _ in range(steps):
            q = self.query_proj(state)[:, None, :]
            attn = torch.softmax((q * keys).sum(-1) / scale, dim=-1)
            context = torch.einsum('bf,bfd->bd', attn, vals)
            state = self.norm(self.update(self.dropout(context), state))
            attns.append(attn)
            states.append(state)
            node_logits.append(self.node_head(state))
            halt_logits.append(self.halt_head(state).squeeze(-1))
        logits = self.out(state)
        if return_trace:
            return logits, {
                'attn': torch.stack(attns, dim=1),
                'states': torch.stack(states, dim=1),
                'node_logits': torch.stack(node_logits, dim=1),
                'halt_logits': torch.stack(halt_logits, dim=1),
            }
        return logits


class NonRecursiveThoughtModel(RecursiveThoughtModel):
    # Same edge encoder, but only one retrieval/update. Tests whether recursion itself matters.
    def forward(self, facts, query, steps=None, return_trace=False):
        return super().forward(facts, query, steps=1, return_trace=return_trace)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for name, model in {
    'baseline': BaselineTransformer(),
    'edge_baseline': EdgeTransformerBaseline(),
    'thought': RecursiveThoughtModel(),
    'nonrecursive_thought': NonRecursiveThoughtModel(),
    'baseline_small': BaselineTransformer(d_model=80, n_layers=2, n_heads=4),
}.items():
    print(name, count_params(model))
"""
    ),
    md("## Training And Evaluation Utilities"),
    code(
        r"""
def accuracy(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()


def forward_model(model, batch, **kwargs):
    if getattr(model, 'uses_depth', False):
        return model(batch.facts, batch.query, batch.depth, **kwargs)
    return model(batch.facts, batch.query, **kwargs)


def thought_step_logits(model, trace):
    return torch.einsum('bsd,nd->bsn', trace['states'][:, 1:], model.entity.weight)


def select_thought_logits(model, batch, trace, answer_mode: str):
    per_step = thought_step_logits(model, trace)
    if answer_mode == 'final':
        return per_step[:, -1]
    if answer_mode == 'depth':
        idx = (batch.depth - 1).clamp(0, per_step.shape[1] - 1)
        return per_step[torch.arange(per_step.shape[0], device=per_step.device), idx]
    if answer_mode == 'halt':
        weights = torch.softmax(trace['halt_logits'], dim=-1)
        return torch.einsum('bs,bsn->bn', weights, per_step)
    raise ValueError(answer_mode)


@torch.no_grad()
def eval_model(model, min_depth, max_depth, *, task='chain', answer_mode='final', batches=CFG['eval_batches'], batch_size=None):
    model.eval()
    batch_size = batch_size or CFG['batch_size']
    total_acc, total_loss, total_n = 0.0, 0.0, 0
    depth_acc, halt_acc, oracle_any_acc = 0.0, 0.0, 0.0
    by_depth = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    by_depth_oracle_depth = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    by_depth_halt = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    for _ in range(batches):
        batch = make_batch(batch_size, min_depth, max_depth, task=task)
        if isinstance(model, RecursiveThoughtModel):
            steps = max(max_depth, getattr(model, 'max_steps', 0))
            _, trace = model(batch.facts, batch.query, steps=steps, return_trace=True)
            per_step = thought_step_logits(model, trace)
            logits = select_thought_logits(model, batch, trace, answer_mode)
            depth_idx = (batch.depth - 1).clamp(0, per_step.shape[1] - 1)
            depth_logits = per_step[torch.arange(per_step.shape[0], device=per_step.device), depth_idx]
            halt_logits = select_thought_logits(model, batch, trace, 'halt')
            depth_pred = depth_logits.argmax(-1)
            halt_pred = halt_logits.argmax(-1)
            oracle_any = (per_step.argmax(-1) == batch.answer[:, None]).any(dim=1)
            depth_acc += (depth_pred == batch.answer).sum().item()
            halt_acc += (halt_pred == batch.answer).sum().item()
            oracle_any_acc += oracle_any.sum().item()
        else:
            logits = forward_model(model, batch)
        loss = F.cross_entropy(logits, batch.answer, reduction='sum').item()
        pred = logits.argmax(-1)
        total_loss += loss
        total_acc += (pred == batch.answer).sum().item()
        total_n += batch.answer.numel()
        for d in range(min_depth, max_depth + 1):
            mask = batch.depth == d
            if mask.any():
                by_depth[d][0] += (pred[mask] == batch.answer[mask]).sum().item()
                by_depth[d][1] += mask.sum().item()
                if isinstance(model, RecursiveThoughtModel):
                    by_depth_oracle_depth[d][0] += (depth_pred[mask] == batch.answer[mask]).sum().item()
                    by_depth_oracle_depth[d][1] += mask.sum().item()
                    by_depth_halt[d][0] += (halt_pred[mask] == batch.answer[mask]).sum().item()
                    by_depth_halt[d][1] += mask.sum().item()
    out = {
        'acc': total_acc / total_n,
        'loss': total_loss / total_n,
        'by_depth': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth.items()},
    }
    if isinstance(model, RecursiveThoughtModel):
        out.update({
            'oracle_depth_acc': depth_acc / total_n,
            'halt_acc': halt_acc / total_n,
            'oracle_any_step_acc': oracle_any_acc / total_n,
            'by_depth_oracle_depth': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth_oracle_depth.items()},
            'by_depth_halt': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth_halt.items()},
        })
    return out


@torch.no_grad()
def eval_thought_stepwise(model, min_depth, max_depth, *, task='chain', batches=CFG['eval_batches']):
    model.eval()
    max_steps = max(max_depth, getattr(model, 'max_steps', CFG['max_steps']))
    step_correct = torch.zeros(max_steps, device=device)
    step_total = 0
    final_correct = 0
    oracle_correct = 0
    by_depth_final = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    by_depth_oracle = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    attn_gold = torch.zeros(max_steps, device=device)
    attn_total = torch.zeros(max_steps, device=device)
    state_delta = torch.zeros(max_steps, device=device)

    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_depth, max_depth, task=task)
        logits, trace = model(batch.facts, batch.query, steps=max_steps, return_trace=True)
        states = trace['states']
        per_step_logits = thought_step_logits(model, trace)
        preds = per_step_logits.argmax(-1)
        correct = preds == batch.answer[:, None]
        final = preds[:, -1]
        depth_idx = (batch.depth - 1).clamp(0, preds.shape[1] - 1)
        depth_pred = preds[torch.arange(preds.shape[0], device=preds.device), depth_idx]
        halt_pred = select_thought_logits(model, batch, trace, 'halt').argmax(-1)
        oracle = correct.any(dim=1)

        step_correct[:correct.shape[1]] += correct.sum(dim=0)
        step_total += correct.shape[0]
        final_correct += (final == batch.answer).sum().item()
        oracle_correct += oracle.sum().item()

        attn = trace['attn']
        for s in range(min(max_steps, attn.shape[1])):
            pos = batch.path_pos[:, s]
            mask = pos >= 0
            if mask.any():
                attn_gold[s] += attn[mask, s].gather(1, pos[mask, None]).sum()
                attn_total[s] += mask.sum()
            state_delta[s] += (states[:, s + 1] - states[:, s]).norm(dim=-1).mean()

        for d in range(min_depth, max_depth + 1):
            mask = batch.depth == d
            if mask.any():
                by_depth_final[d][0] += (final[mask] == batch.answer[mask]).sum().item()
                by_depth_final[d][1] += mask.sum().item()
                by_depth_oracle[d][0] += (depth_pred[mask] == batch.answer[mask]).sum().item()
                by_depth_oracle[d][1] += mask.sum().item()

    denom = batches
    return {
        'final_accuracy': final_correct / step_total,
        'oracle_accuracy': oracle_correct / step_total,
        'learned_halt_accuracy': eval_model(model, min_depth, max_depth, task=task, answer_mode='halt', batches=batches)['acc'],
        'step_accuracy': (step_correct / step_total).detach().cpu().tolist(),
        'gold_attention_by_step': (attn_gold / attn_total.clamp_min(1)).detach().cpu().tolist(),
        'state_delta_by_step': (state_delta / denom).detach().cpu().tolist(),
        'by_depth_final': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth_final.items()},
        'by_depth_oracle': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth_oracle.items()},
    }


def aux_losses(model, batch, trace, mode: str, aux_weight: float):
    if aux_weight <= 0 or mode == 'none':
        return torch.tensor(0.0, device=device), {}
    losses, metrics = [], {}
    if mode in ('attention', 'attention_plus_node', 'annealed_attention', 'attention_plus_halt'):
        attn = trace['attn']
        B, S, F_ = attn.shape
        pos = batch.path_pos[:, :S]
        mask = pos >= 0
        if mask.any():
            attn_logits = (attn.clamp_min(1e-8)).log()
            attn_loss = F.nll_loss(attn_logits[mask], pos[mask])
            losses.append(attn_loss)
            metrics['attn_loss'] = float(attn_loss.detach().cpu())
    if mode in ('node', 'attention_plus_node'):
        node_logits = trace['node_logits']
        target_nodes = batch.path_nodes[:, 1:1 + node_logits.shape[1]]
        mask = target_nodes >= 0
        if mask.any():
            node_loss = F.cross_entropy(node_logits[mask], target_nodes[mask])
            losses.append(node_loss)
            metrics['node_loss'] = float(node_loss.detach().cpu())
    if mode in ('halt', 'attention_plus_halt'):
        halt_target = (batch.depth - 1).clamp(0, trace['halt_logits'].shape[1] - 1)
        halt_loss = F.cross_entropy(trace['halt_logits'], halt_target)
        losses.append(halt_loss)
        metrics['halt_loss'] = float(halt_loss.detach().cpu())
    if not losses:
        return torch.tensor(0.0, device=device), metrics
    total = sum(losses) * aux_weight
    metrics['aux_loss'] = float(total.detach().cpu())
    return total, metrics


def current_aux_weight(mode: str, base_weight: float, step: int, steps: int):
    if mode == 'annealed_attention':
        return base_weight * max(0.0, 1.0 - step / max(1, int(0.8 * steps)))
    return base_weight


def answer_mode_for_step(mode: str, step: int, steps: int):
    if mode == 'depth_then_halt':
        return 'depth' if step <= max(1, steps // 2) else 'halt'
    return mode


def train_depth_for_step(step: int, steps: int, schedule: str):
    if schedule == 'fixed':
        return TRAIN_DEPTH
    if schedule == 'curriculum':
        first = max(1, steps // 3)
        second = max(first + 1, 2 * steps // 3)
        if step <= first:
            return (1, 3)
        if step <= second:
            return (1, min(6, CFG['max_steps']))
        return (1, CFG['max_steps'])
    if schedule == 'full':
        return (1, CFG['max_steps'])
    raise ValueError(schedule)


def train_model(
    model,
    name: str,
    *,
    steps: int,
    task: str = 'chain',
    seed: int = 1,
    aux_mode: str = 'none',
    aux_weight: float = 0.5,
    answer_mode: str = 'final',
    eval_answer_mode: Optional[str] = None,
    depth_schedule: str = 'fixed',
    log_every: Optional[int] = None,
):
    set_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    log_every = log_every or max(1, steps // 4)
    eval_answer_mode = eval_answer_mode or ('halt' if answer_mode == 'depth_then_halt' else answer_mode)
    history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        train_depth = train_depth_for_step(step, steps, depth_schedule)
        batch = make_batch(CFG['batch_size'], *train_depth, task=task)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            step_answer_mode = answer_mode_for_step(answer_mode, step, steps)
            if aux_mode == 'none' and step_answer_mode == 'final':
                logits = forward_model(model, batch)
                extra = {}
                aux = torch.tensor(0.0, device=device)
            elif isinstance(model, RecursiveThoughtModel):
                logits, trace = model(batch.facts, batch.query, return_trace=True)
                logits = select_thought_logits(model, batch, trace, step_answer_mode)
                w = current_aux_weight(aux_mode, aux_weight, step, steps)
                aux, extra = aux_losses(model, batch, trace, aux_mode, w)
            else:
                logits = forward_model(model, batch)
                extra = {}
                aux = torch.tensor(0.0, device=device)
            answer_loss = F.cross_entropy(logits, batch.answer)
            loss = answer_loss + aux
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % log_every == 0 or step == steps:
            row = {
                'step': step,
                'train_loss': float(loss.detach().cpu()),
                'answer_loss': float(answer_loss.detach().cpu()),
                'train_acc': accuracy(logits.detach(), batch.answer),
                'aux_weight': current_aux_weight(aux_mode, aux_weight, step, steps),
                'answer_mode': answer_mode,
                'step_answer_mode': step_answer_mode,
                'depth_schedule': depth_schedule,
                'train_depth': train_depth,
                'elapsed_s': time.time() - t0,
            }
            row.update(extra)
            row['iid_acc'] = eval_model(model, *TRAIN_DEPTH, task=task, answer_mode=eval_answer_mode, batches=max(1, CFG['eval_batches'] // 4))['acc']
            row['ood_acc'] = eval_model(model, *OOD_DEPTH, task=task, answer_mode=eval_answer_mode, batches=max(1, CFG['eval_batches'] // 4))['acc']
            history.append(row)
            print(name, row)
    return history


@torch.no_grad()
def benchmark_model(model, *, task='chain', batches=20):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], *OOD_DEPTH, task=task)
        _ = forward_model(model, batch)
        n += batch.answer.numel()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {'examples_per_s': n / elapsed, 'ms_per_1k_examples': elapsed * 1000 / max(1, n) * 1000}
"""
    ),
    md("## Experiment Matrix\n\nThis matrix is intentionally broader than the earlier notebook. It includes baselines, recursion ablations, supervised intermediate-thought variants, annealed supervision, harder tasks, and multiple seeds."),
    code(
        r"""
def build_model(kind: str):
    if kind == 'baseline':
        return BaselineTransformer()
    if kind == 'edge_baseline':
        return EdgeTransformerBaseline()
    if kind == 'edge_depth_baseline':
        return EdgeTransformerBaseline(use_depth=True, max_depth=CFG['max_steps'] + 8)
    if kind == 'baseline_small':
        return BaselineTransformer(d_model=80, n_layers=2, n_heads=4)
    if kind == 'baseline_large':
        return BaselineTransformer(d_model=160, n_layers=4, n_heads=4)
    if kind == 'thought':
        return RecursiveThoughtModel()
    if kind == 'thought_large':
        return RecursiveThoughtModel(d_model=192)
    if kind == 'thought_nonrecursive':
        return NonRecursiveThoughtModel()
    raise ValueError(kind)


EXPERIMENTS = [
    # Main no-selfloop baselines. Depth-token baseline is the strongest flat comparison.
    dict(name='baseline_edge_chain_fixed', kind='edge_baseline', task='chain', aux_mode='none', answer_mode='final', depth_schedule='fixed', steps=CFG['train_steps']),
    dict(name='baseline_edge_depth_chain_fixed', kind='edge_depth_baseline', task='chain', aux_mode='none', answer_mode='final', depth_schedule='fixed', steps=CFG['train_steps']),
    dict(name='baseline_edge_depth_chain_curriculum', kind='edge_depth_baseline', task='chain', aux_mode='none', answer_mode='final', depth_schedule='curriculum', steps=CFG['aux_steps']),

    # Core recursive tests without self-loop shortcuts.
    dict(name='thought_depth_oracle_terminal_only', kind='thought', task='chain', aux_mode='none', answer_mode='depth', depth_schedule='fixed', steps=CFG['train_steps']),
    dict(name='thought_attention_depth_oracle', kind='thought', task='chain', aux_mode='attention', aux_weight=0.5, answer_mode='depth', depth_schedule='fixed', steps=CFG['aux_steps']),
    dict(name='thought_attention_depth_curriculum', kind='thought', task='chain', aux_mode='attention', aux_weight=0.5, answer_mode='depth', depth_schedule='curriculum', steps=CFG['aux_steps']),

    # Learned stopping. The two-stage version first trains composition with oracle-depth answers, then switches to halting.
    dict(name='thought_attention_halt_curriculum', kind='thought', task='chain', aux_mode='attention_plus_halt', aux_weight=0.35, answer_mode='halt', depth_schedule='curriculum', steps=CFG['aux_steps']),
    dict(name='thought_attention_halt_twostage_curriculum', kind='thought', task='chain', aux_mode='attention_plus_halt', aux_weight=0.35, answer_mode='depth_then_halt', eval_answer_mode='halt', depth_schedule='curriculum', steps=CFG['aux_steps']),

    # Self-loop is now diagnostic only: it tests answer-retention/stopping confounds, not reasoning proof.
    dict(name='diagnostic_baseline_edge_chain_selfloop', kind='edge_baseline', task='chain_selfloop', aux_mode='none', answer_mode='final', depth_schedule='fixed', steps=CFG['sweep_steps']),
    dict(name='diagnostic_thought_selfloop_terminal_only', kind='thought', task='chain_selfloop', aux_mode='none', answer_mode='final', depth_schedule='fixed', steps=CFG['sweep_steps']),
    dict(name='thought_annealed_attention_selfloop', kind='thought', task='chain_selfloop', aux_mode='annealed_attention', aux_weight=0.7, answer_mode='final', depth_schedule='fixed', steps=CFG['aux_steps']),

    # Harder structural tests without self-loop shortcuts.
    dict(name='baseline_edge_depth_multirelation_curriculum', kind='edge_depth_baseline', task='multirelation', aux_mode='none', answer_mode='final', depth_schedule='curriculum', steps=CFG['aux_steps']),
    dict(name='thought_attention_depth_multirelation_curriculum', kind='thought', task='multirelation', aux_mode='attention', aux_weight=0.5, answer_mode='depth', depth_schedule='curriculum', steps=CFG['aux_steps']),
    dict(name='baseline_edge_depth_branch_trap_curriculum', kind='edge_depth_baseline', task='branch_trap', aux_mode='none', answer_mode='final', depth_schedule='curriculum', steps=CFG['aux_steps']),
    dict(name='thought_attention_depth_branch_trap_curriculum', kind='thought', task='branch_trap', aux_mode='attention', aux_weight=0.5, answer_mode='depth', depth_schedule='curriculum', steps=CFG['aux_steps']),
]

# One seed for heavier variants, all seeds for core conclusion.
MULTISEED_NAMES = {
    'baseline_edge_chain_fixed',
    'baseline_edge_depth_chain_fixed',
    'baseline_edge_depth_chain_curriculum',
    'thought_depth_oracle_terminal_only',
    'thought_attention_depth_curriculum',
    'thought_attention_halt_curriculum',
    'thought_attention_halt_twostage_curriculum',
}

if FAST_RUN:
    EXPERIMENTS = EXPERIMENTS[:5]

print('experiments:', len(EXPERIMENTS))
for exp in EXPERIMENTS:
    seeds = CFG['seeds'] if exp['name'] in MULTISEED_NAMES else [CFG['seeds'][0]]
    print(exp['name'], 'seeds=', seeds, 'steps=', exp['steps'], 'task=', exp['task'], 'aux=', exp['aux_mode'])
"""
    ),
    md("## Run Experiments"),
    code(
        r"""
results = {}
trained_models = {}

global_t0 = time.time()
for exp in EXPERIMENTS:
    seeds = CFG['seeds'] if exp['name'] in MULTISEED_NAMES else [CFG['seeds'][0]]
    for seed in seeds:
        run_name = f"{exp['name']}_seed{seed}"
        print('\\n===', run_name, '===')
        model = build_model(exp['kind'])
        hist = train_model(
            model,
            run_name,
            steps=exp['steps'],
            task=exp['task'],
            seed=seed,
            aux_mode=exp.get('aux_mode', 'none'),
            aux_weight=exp.get('aux_weight', 0.5),
            answer_mode=exp.get('answer_mode', 'final'),
            eval_answer_mode=exp.get('eval_answer_mode'),
            depth_schedule=exp.get('depth_schedule', 'fixed'),
        )
        answer_mode = exp.get('eval_answer_mode', exp.get('answer_mode', 'final'))
        iid = eval_model(model, *TRAIN_DEPTH, task=exp['task'], answer_mode=answer_mode)
        ood = eval_model(model, *OOD_DEPTH, task=exp['task'], answer_mode=answer_mode)
        hard = eval_model(model, *HARD_DEPTH, task=exp['task'], answer_mode=answer_mode, batches=max(4, CFG['eval_batches'] // 2))
        bench = benchmark_model(model, task=exp['task'], batches=5 if FAST_RUN else 15)
        row = {
            'spec': exp,
            'seed': seed,
            'params': count_params(model),
            'history': hist,
            'iid': iid,
            'ood': ood,
            'hard_ood': hard,
            'benchmark': bench,
            'elapsed_s_total': time.time() - global_t0,
        }
        if isinstance(model, RecursiveThoughtModel):
            row['step_diagnostics'] = eval_thought_stepwise(model, 1, CFG['max_steps'], task=exp['task'])
        results[run_name] = row
        trained_models[run_name] = model
        print(json.dumps({k: row[k] for k in ['params', 'iid', 'ood', 'hard_ood', 'benchmark']}, indent=2))

print('total elapsed seconds:', time.time() - global_t0)
"""
    ),
    md("## Summaries And Hypothesis Tests"),
    code(
        r"""
def group_runs(results):
    groups = {}
    for run_name, row in results.items():
        base = row['spec']['name']
        groups.setdefault(base, []).append(row)
    return groups


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {'mean': None, 'std': None, 'n': 0}
    return {
        'mean': statistics.mean(vals),
        'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
        'n': len(vals),
    }


groups = group_runs(results)
summary = {}
for name, rows in groups.items():
    params_vals = [r['params'] for r in rows]
    ood_vals = [r['ood']['acc'] for r in rows]
    hard_vals = [r['hard_ood']['acc'] for r in rows]
    summary[name] = {
        'params': mean_std(params_vals),
        'iid_acc': mean_std([r['iid']['acc'] for r in rows]),
        'ood_acc': mean_std(ood_vals),
        'ood_oracle_depth_acc': mean_std([r['ood'].get('oracle_depth_acc') for r in rows if 'oracle_depth_acc' in r['ood']]),
        'ood_halt_acc': mean_std([r['ood'].get('halt_acc') for r in rows if 'halt_acc' in r['ood']]),
        'ood_oracle_any_step_acc': mean_std([r['ood'].get('oracle_any_step_acc') for r in rows if 'oracle_any_step_acc' in r['ood']]),
        'hard_ood_acc': mean_std(hard_vals),
        'ood_per_million_params': mean_std([acc / (params / 1_000_000) for acc, params in zip(ood_vals, params_vals)]),
        'hard_ood_per_million_params': mean_std([acc / (params / 1_000_000) for acc, params in zip(hard_vals, params_vals)]),
        'examples_per_s': mean_std([r['benchmark']['examples_per_s'] for r in rows]),
        'task': rows[0]['spec']['task'],
        'aux_mode': rows[0]['spec']['aux_mode'],
        'answer_mode': rows[0]['spec'].get('answer_mode', 'final'),
        'eval_answer_mode': rows[0]['spec'].get('eval_answer_mode', rows[0]['spec'].get('answer_mode', 'final')),
        'depth_schedule': rows[0]['spec'].get('depth_schedule', 'fixed'),
    }

print(json.dumps(summary, indent=2))


def get_mean(name, metric='ood_acc'):
    return summary.get(name, {}).get(metric, {}).get('mean')


def verdict(label, condition, support_text, against_text):
    return {
        'hypothesis': label,
        'supported': bool(condition),
        'interpretation': support_text if condition else against_text,
    }


hypotheses = []
hypotheses.append(verdict(
    'H1 no_selfloop_architecture_advantage',
    (get_mean('thought_attention_depth_curriculum') or 0) > (get_mean('baseline_edge_depth_chain_curriculum') or 0) + 0.05,
    'Recursive thought beats a depth-aware edge-token Transformer on the clean no-selfloop task.',
    'Depth-aware edge-token Transformer is comparable or better on the clean no-selfloop task.',
))
hypotheses.append(verdict(
    'H2 parameter_efficiency_no_selfloop',
    (get_mean('thought_attention_depth_curriculum', 'ood_per_million_params') or 0) > (get_mean('baseline_edge_depth_chain_curriculum', 'ood_per_million_params') or 0) + 0.25,
    'Recursive thought gives more OOD accuracy per parameter than the depth-aware edge baseline.',
    'Recursive thought does not show clear parameter-efficiency over the depth-aware edge baseline.',
))
hypotheses.append(verdict(
    'H3 intermediate_supervision_helps',
    (get_mean('thought_attention_depth_curriculum') or 0) > (get_mean('thought_depth_oracle_terminal_only') or 0) + 0.10,
    'Intermediate attention supervision substantially improves recursive composition.',
    'Intermediate attention supervision does not substantially improve recursive composition.',
))
hypotheses.append(verdict(
    'H4 curriculum_for_hard_depth',
    (get_mean('thought_attention_depth_curriculum', 'hard_ood_acc') or 0) > (get_mean('thought_attention_depth_oracle', 'hard_ood_acc') or 0) + 0.20,
    'Depth curriculum materially improves extrapolation to hard depths 11-14.',
    'Depth curriculum does not materially improve hard-depth extrapolation.',
))
hypotheses.append(verdict(
    'H5 learned_halting',
    (get_mean('thought_attention_halt_twostage_curriculum') or 0) > (get_mean('thought_attention_halt_curriculum') or 0) + 0.03,
    'Two-stage learned halting improves over direct learned halting.',
    'Two-stage learned halting does not improve over direct learned halting.',
))
hypotheses.append(verdict(
    'H6 harder_structure_survives_no_selfloop',
    (get_mean('thought_attention_depth_multirelation_curriculum') or 0) > (get_mean('baseline_edge_depth_multirelation_curriculum') or 0) + 0.05
    and (get_mean('thought_attention_depth_branch_trap_curriculum') or 0) > (get_mean('baseline_edge_depth_branch_trap_curriculum') or 0) + 0.05,
    'The recursive thought advantage survives multi-relation and branch-trap tasks without self-loop shortcuts.',
    'The advantage does not clearly survive harder structural variants.',
))
hypotheses.append(verdict(
    'H7 selfloop_is_only_diagnostic',
    (get_mean('diagnostic_baseline_edge_chain_selfloop') or 0) > 0.90 and (get_mean('diagnostic_thought_selfloop_terminal_only') or 0) > 0.90,
    'Self-loop is easy for both model families and should be treated only as a stopping-confound diagnostic.',
    'Self-loop is not uniformly easy, so it may still be informative beyond stopping diagnostics.',
))

print(json.dumps(hypotheses, indent=2))
"""
    ),
    md("## Plots"),
    code(
        r"""
def plot_summary(summary):
    names = list(summary.keys())
    ood = [summary[n]['ood_acc']['mean'] for n in names]
    hard = [summary[n]['hard_ood_acc']['mean'] for n in names]
    params = [summary[n]['params']['mean'] for n in names]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(max(10, len(names) * 0.8), 5))
    ax1.bar(x - 0.18, ood, width=0.36, label='OOD depth')
    ax1.bar(x + 0.18, hard, width=0.36, label='Harder OOD')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(x, params, color='black', marker='o', linestyle='--', label='params')
    ax2.set_ylabel('parameters')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_depth_curves(results):
    plt.figure(figsize=(10, 5))
    for name, row in results.items():
        if not name.endswith(f"seed{CFG['seeds'][0]}"):
            continue
        bd = row['ood']['by_depth']
        xs = [int(k) for k, v in bd.items() if v is not None]
        ys = [bd[str(k)] for k in xs]
        plt.plot(xs, ys, marker='o', label=row['spec']['name'])
    plt.xlabel('OOD depth')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.05)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


plot_summary(summary)
plot_depth_curves(results)
"""
    ),
    md("## Inspect A Thought Trace\n\nThis cell helps verify that recursive states actually change and attend to gold facts. Strong final accuracy without gold attention would be less evidence for the intended mechanism."),
    code(
        r"""
best_name = None
for preferred in ['thought_attention_halt_twostage_curriculum_seed1', 'thought_attention_halt_curriculum_seed1', 'thought_attention_depth_curriculum_seed1', 'thought_depth_oracle_terminal_only_seed1']:
    if preferred in trained_models:
        best_name = preferred
        break

if best_name:
    model = trained_models[best_name]
    model.eval()
    batch = make_batch(1, CFG['max_steps'], CFG['max_steps'], task=results[best_name]['spec']['task'])
    logits, trace = model(batch.facts, batch.query, return_trace=True)
    describe_batch(batch, 0)
    print('prediction:', int(logits.argmax(-1)[0]), 'model:', best_name)
    for s in range(trace['attn'].shape[1]):
        idx = int(trace['attn'][0, s].argmax())
        gold = int(batch.path_pos[0, s]) if int(batch.path_pos[0, s]) >= 0 else None
        print({
            'step': s + 1,
            'chosen_fact_index': idx,
            'gold_fact_index': gold,
            'chosen_attention': float(trace['attn'][0, s, idx]),
            'gold_attention': float(trace['attn'][0, s, gold]) if gold is not None else None,
            'chosen_edge': batch.facts[0, idx].detach().cpu().tolist(),
        })
else:
    print('No recursive model found.')
"""
    ),
    md("## Save Report\n\nThe JSON report is designed to be pasted back into Codex for analysis. It includes raw histories, grouped summaries, verdicts, depth curves, diagnostics, and benchmark measurements."),
    code(
        r"""
report = {
    'version': 'ultimate_v3_no_selfloop_depth_baseline',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'profile': PROFILE,
    'fast_run': FAST_RUN,
    'device': str(device),
    'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'config': {
        **CFG,
        'lr': LR,
        'n_relations': N_RELATIONS,
        'train_depth': TRAIN_DEPTH,
        'ood_depth': OOD_DEPTH,
        'hard_depth': HARD_DEPTH,
    },
    'summary': summary,
    'hypotheses': hypotheses,
    'results': results,
}

out = Path('/kaggle/working/recursive_thought_tokens_ultimate_v3_report.json') if Path('/kaggle/working').exists() else Path('recursive_thought_tokens_ultimate_v3_report.json')
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('saved', out)
"""
    ),
    md(
        r"""
## How To Read The Result

The strongest support for the original idea is the combination of:

1. `thought_attention_depth_curriculum` beating `baseline_edge_depth_chain_curriculum`: evidence for an architecture advantage over a depth-aware flat Transformer.
2. `thought_attention_depth_curriculum` beating `thought_depth_oracle_terminal_only`: evidence that intermediate thought supervision matters.
3. Hard-depth gains over `thought_attention_depth_oracle`: evidence that curriculum improves depth extrapolation.
4. `thought_attention_halt_twostage_curriculum` working: evidence that learned stopping can replace oracle depth.
5. Hard task wins on no-selfloop `multirelation` and `branch_trap`: evidence the method is not just memorizing a simple chain algorithm.
6. Self-loop diagnostics being easy for both families: evidence that self-loop should not be used as the main proof.

The idea is weakened if the depth-aware edge-token baseline catches up, if learned halting collapses, or if no-selfloop harder tasks collapse.
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/kaggle_recursive_thought_tokens_proof.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
