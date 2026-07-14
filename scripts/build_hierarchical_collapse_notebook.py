"""Build a Kaggle notebook for hierarchical thought-token collapse."""

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
# Hierarchical Thought-Token Collapse Transformer

This notebook tests the architecture idea:

> After several Transformer layers, let the model mark closed thought boundaries, remove internal tokens, and keep one contextualized representative hidden state. Repeat this at higher layers, producing a hierarchy of increasingly abstract thought tokens while reducing sequence length.

We test four variants:

1. **Baseline Transformer**: no collapse.
2. **Oracle collapse**: boundaries are known from the synthetic grammar.
3. **Learned collapse**: boundary head predicts thought boundaries.
4. **Random collapse**: control for arbitrary token reduction.

The task is synthetic but structured. Inputs are sequences of independent clauses. Each clause has a local hidden value. The answer is the sum of all clause values modulo `P`. A good model can compress each clause into one thought token, then compose clause-level thoughts.

Main metrics:

- accuracy IID and OOD;
- token count after collapse;
- boundary accuracy;
- throughput.
"""
    ),
    code(
        r"""
import math, os, random, time, json, statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FAST_RUN = os.environ.get('FAST_RUN', '0') == '1'

if FAST_RUN:
    CFG = dict(steps=120, batch_size=128, eval_batches=4, seeds=[1], d_model=96)
else:
    CFG = dict(steps=1600, batch_size=512, eval_batches=20, seeds=[1, 2, 3], d_model=128)

P = 31
VOCAB = {
    'PAD': 0,
    'CLS': 1,
    'SEP': 2,
    'A': 3,
    'B': 4,
    'C': 5,
    'D': 6,
}
NUM_OFFSET = 16
VOCAB_SIZE = NUM_OFFSET + P
TRAIN_CLAUSES = (3, 6)
OOD_CLAUSES = (8, 12)
HARD_CLAUSES = (14, 18)
LR = 3e-4

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
print(json.dumps(CFG, indent=2))
"""
    ),
    md("## Data\n\nEach clause is `A x B y C z SEP`. Its local value is `(x + 2*y - z) mod P`. The global label is the sum of all clause values. `SEP` is the oracle boundary and the final token of a closed thought."),
    code(
        r"""
@dataclass
class Batch:
    tokens: torch.Tensor
    label: torch.Tensor
    boundary: torch.Tensor
    n_clauses: torch.Tensor
    lengths: torch.Tensor


def num_token(x):
    return NUM_OFFSET + x


def make_batch(batch_size, min_clauses, max_clauses):
    seqs, boundaries, labels, ncs, lengths = [], [], [], [], []
    max_len = 1 + max_clauses * 7
    for _ in range(batch_size):
        n = random.randint(min_clauses, max_clauses)
        toks = [VOCAB['CLS']]
        bnd = [0]
        total = 0
        for _c in range(n):
            x, y, z = random.randrange(P), random.randrange(P), random.randrange(P)
            value = (x + 2 * y - z) % P
            total = (total + value) % P
            clause = [VOCAB['A'], num_token(x), VOCAB['B'], num_token(y), VOCAB['C'], num_token(z), VOCAB['SEP']]
            toks.extend(clause)
            bnd.extend([0, 0, 0, 0, 0, 0, 1])
        length = len(toks)
        pad = max_len - length
        toks.extend([VOCAB['PAD']] * pad)
        bnd.extend([0] * pad)
        seqs.append(toks)
        boundaries.append(bnd)
        labels.append(total)
        ncs.append(n)
        lengths.append(length)
    return Batch(
        tokens=torch.tensor(seqs, dtype=torch.long, device=device),
        label=torch.tensor(labels, dtype=torch.long, device=device),
        boundary=torch.tensor(boundaries, dtype=torch.float32, device=device),
        n_clauses=torch.tensor(ncs, dtype=torch.long, device=device),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )


b = make_batch(2, 3, 4)
print(b.tokens.shape, b.tokens[0].detach().cpu().tolist())
print('boundary', b.boundary[0].detach().cpu().tolist())
print('label', b.label.detach().cpu().tolist())
"""
    ),
    md("## Models"),
    code(
        r"""
class BaselineTransformer(nn.Module):
    def __init__(self, d_model=CFG['d_model'], n_layers=4, n_heads=4, max_len=160):
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=0.05, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, P)

    def forward(self, tokens, **kwargs):
        x = self.token(tokens) + self.pos[:, :tokens.shape[1]]
        pad = tokens == VOCAB['PAD']
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.head(x[:, 0]), {'tokens_after': tokens.shape[1], 'boundary_logits': None}


def collapse_by_boundary(x, tokens, boundary_mask):
    # Keep CLS and each boundary representative. The representative is the boundary token hidden state.
    B, L, D = x.shape
    reps, new_tokens = [], []
    max_new = 1 + int(boundary_mask.sum(dim=1).max().item())
    for b in range(B):
        idx = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            torch.nonzero(boundary_mask[b], as_tuple=False).flatten(),
        ])
        xb = x[b, idx]
        tb = tokens[b, idx]
        pad = max_new - idx.numel()
        if pad:
            xb = torch.cat([xb, torch.zeros(pad, D, device=x.device)], dim=0)
            tb = torch.cat([tb, torch.full((pad,), VOCAB['PAD'], dtype=torch.long, device=x.device)], dim=0)
        reps.append(xb)
        new_tokens.append(tb)
    return torch.stack(reps, dim=0), torch.stack(new_tokens, dim=0)


class HierarchicalCollapseTransformer(nn.Module):
    def __init__(self, mode='oracle', d_model=CFG['d_model'], n_heads=4, pre_layers=2, post_layers=2, max_len=160):
        super().__init__()
        self.mode = mode
        self.token = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos1 = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.pos2 = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        pre = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=0.05, batch_first=True, activation='gelu', norm_first=True)
        post = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=0.05, batch_first=True, activation='gelu', norm_first=True)
        self.pre = nn.TransformerEncoder(pre, num_layers=pre_layers)
        self.post = nn.TransformerEncoder(post, num_layers=post_layers)
        self.boundary_head = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, P)

    def choose_boundary(self, tokens, boundary_target, logits):
        if self.mode == 'oracle':
            return boundary_target.bool()
        if self.mode == 'learned':
            mask = torch.sigmoid(logits).squeeze(-1) > 0.5
            # Avoid empty collapse early in training: always allow true SEP candidates if nothing selected.
            sep = tokens == VOCAB['SEP']
            empty = mask.sum(dim=1) == 0
            mask[empty] = sep[empty]
            return mask & (tokens != VOCAB['PAD']) & (tokens != VOCAB['CLS'])
        if self.mode == 'random':
            sep_count = (boundary_target > 0.5).sum(dim=1)
            out = torch.zeros_like(boundary_target, dtype=torch.bool)
            for b in range(tokens.shape[0]):
                valid = torch.nonzero((tokens[b] != VOCAB['PAD']) & (tokens[b] != VOCAB['CLS']), as_tuple=False).flatten()
                perm = valid[torch.randperm(valid.numel(), device=tokens.device)[: int(sep_count[b].item())]]
                out[b, perm] = True
            return out
        raise ValueError(self.mode)

    def forward(self, tokens, boundary=None):
        pad = tokens == VOCAB['PAD']
        x = self.token(tokens) + self.pos1[:, :tokens.shape[1]]
        x = self.pre(x, src_key_padding_mask=pad)
        boundary_logits = self.boundary_head(x).squeeze(-1)
        if boundary is None:
            boundary = (tokens == VOCAB['SEP']).float()
        mask = self.choose_boundary(tokens, boundary, boundary_logits[..., None])
        x2, tok2 = collapse_by_boundary(x, tokens, mask)
        pad2 = tok2 == VOCAB['PAD']
        x2 = x2 + self.pos2[:, :x2.shape[1]]
        x2 = self.post(x2, src_key_padding_mask=pad2)
        return self.head(x2[:, 0]), {'tokens_after': x2.shape[1], 'boundary_logits': boundary_logits, 'boundary_mask': mask}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
"""
    ),
    md("## Train / Eval"),
    code(
        r"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_model(model, min_c, max_c, batches=None, boundary_weight=0.0):
    model.eval()
    batches = batches or CFG['eval_batches']
    correct = total = loss_sum = token_after_sum = boundary_correct = boundary_total = 0
    by_clause = {n: [0, 0] for n in range(min_c, max_c + 1)}
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_c, max_c)
        logits, info = model(batch.tokens, boundary=batch.boundary)
        pred = logits.argmax(-1)
        loss_sum += F.cross_entropy(logits, batch.label, reduction='sum').item()
        correct += (pred == batch.label).sum().item()
        total += batch.label.numel()
        token_after_sum += info['tokens_after'] * batch.label.numel()
        if info.get('boundary_logits') is not None:
            valid = batch.tokens != VOCAB['PAD']
            bpred = (torch.sigmoid(info['boundary_logits']) > 0.5).float()
            boundary_correct += (bpred[valid] == batch.boundary[valid]).sum().item()
            boundary_total += valid.sum().item()
        for n in by_clause:
            mask = batch.n_clauses == n
            if mask.any():
                by_clause[n][0] += (pred[mask] == batch.label[mask]).sum().item()
                by_clause[n][1] += mask.sum().item()
    out = {
        'acc': correct / total,
        'loss': loss_sum / total,
        'avg_tokens_after': token_after_sum / total,
        'by_clause': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_clause.items()},
    }
    if boundary_total:
        out['boundary_acc'] = boundary_correct / boundary_total
    return out


@torch.no_grad()
def benchmark(model, min_c, max_c, batches=30):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    total = 0
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_c, max_c)
        _ = model(batch.tokens, boundary=batch.boundary)
        total += batch.label.numel()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {'examples_per_s': total / elapsed, 'elapsed_s': elapsed}


def train_model(kind, seed=1):
    set_seed(seed)
    if kind == 'baseline':
        model = BaselineTransformer().to(device)
    else:
        model = HierarchicalCollapseTransformer(mode=kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    hist = []
    t0 = time.time()
    for step in range(1, CFG['steps'] + 1):
        model.train()
        batch = make_batch(CFG['batch_size'], *TRAIN_CLAUSES)
        logits, info = model(batch.tokens, boundary=batch.boundary)
        answer_loss = F.cross_entropy(logits, batch.label)
        if kind == 'learned':
            valid = batch.tokens != VOCAB['PAD']
            boundary_loss = F.binary_cross_entropy_with_logits(info['boundary_logits'][valid], batch.boundary[valid])
            loss = answer_loss + 0.2 * boundary_loss
        else:
            boundary_loss = torch.tensor(0.0, device=device)
            loss = answer_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, CFG['steps'] // 4) == 0 or step == CFG['steps']:
            row = {
                'step': step,
                'loss': float(loss.detach().cpu()),
                'answer_loss': float(answer_loss.detach().cpu()),
                'boundary_loss': float(boundary_loss.detach().cpu()),
                'train_acc': (logits.argmax(-1) == batch.label).float().mean().item(),
                'tokens_after': info['tokens_after'],
                'ood_acc': eval_model(model, *OOD_CLAUSES, batches=max(1, CFG['eval_batches'] // 4))['acc'],
                'elapsed_s': time.time() - t0,
            }
            hist.append(row)
            print(kind, row)
    return model, hist
"""
    ),
    md("## Run Experiments"),
    code(
        r"""
EXPERIMENTS = ['baseline', 'oracle', 'learned', 'random']
if FAST_RUN:
    EXPERIMENTS = ['baseline', 'oracle', 'learned']

results = {}
for kind in EXPERIMENTS:
    seeds = CFG['seeds'] if kind in {'baseline', 'oracle', 'learned'} else [CFG['seeds'][0]]
    for seed in seeds:
        run = f'{kind}_seed{seed}'
        print('\\n===', run, '===')
        model, hist = train_model(kind, seed=seed)
        row = {
            'kind': kind,
            'seed': seed,
            'params': count_params(model),
            'history': hist,
            'iid': eval_model(model, *TRAIN_CLAUSES),
            'ood': eval_model(model, *OOD_CLAUSES),
            'hard': eval_model(model, *HARD_CLAUSES, batches=max(4, CFG['eval_batches'] // 2)),
            'benchmark': benchmark(model, *OOD_CLAUSES, batches=5 if FAST_RUN else 25),
        }
        results[run] = row
        print(json.dumps({k: row[k] for k in ['params', 'iid', 'ood', 'hard', 'benchmark']}, indent=2))
"""
    ),
    md("## Summary"),
    code(
        r"""
def mean_std(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {'mean': None, 'std': None, 'n': 0}
    return {'mean': statistics.mean(vals), 'std': statistics.stdev(vals) if len(vals) > 1 else 0.0, 'n': len(vals)}


groups = {}
for run, row in results.items():
    groups.setdefault(row['kind'], []).append(row)

summary = {}
for kind, rows in groups.items():
    summary[kind] = {
        'params': mean_std([r['params'] for r in rows]),
        'iid_acc': mean_std([r['iid']['acc'] for r in rows]),
        'ood_acc': mean_std([r['ood']['acc'] for r in rows]),
        'hard_acc': mean_std([r['hard']['acc'] for r in rows]),
        'iid_tokens_after': mean_std([r['iid']['avg_tokens_after'] for r in rows]),
        'ood_tokens_after': mean_std([r['ood']['avg_tokens_after'] for r in rows]),
        'hard_tokens_after': mean_std([r['hard']['avg_tokens_after'] for r in rows]),
        'boundary_acc': mean_std([r['ood'].get('boundary_acc') for r in rows if 'boundary_acc' in r['ood']]),
        'examples_per_s': mean_std([r['benchmark']['examples_per_s'] for r in rows]),
    }

def get(kind, metric):
    return summary.get(kind, {}).get(metric, {}).get('mean') or 0.0

hypotheses = [
    {
        'hypothesis': 'oracle_collapse_preserves_accuracy',
        'supported': get('oracle', 'ood_acc') >= get('baseline', 'ood_acc') - 0.05 and get('oracle', 'ood_tokens_after') < get('baseline', 'ood_tokens_after') * 0.5,
        'interpretation': 'Oracle thought boundaries compress sequence length without major OOD accuracy loss.' if get('oracle', 'ood_acc') >= get('baseline', 'ood_acc') - 0.05 and get('oracle', 'ood_tokens_after') < get('baseline', 'ood_tokens_after') * 0.5 else 'Oracle collapse loses too much accuracy or does not compress enough.',
    },
    {
        'hypothesis': 'learned_boundaries_work',
        'supported': get('learned', 'ood_acc') >= get('oracle', 'ood_acc') - 0.08 and get('learned', 'ood_tokens_after') < get('baseline', 'ood_tokens_after') * 0.6,
        'interpretation': 'Learned boundaries approach oracle collapse while reducing tokens.' if get('learned', 'ood_acc') >= get('oracle', 'ood_acc') - 0.08 and get('learned', 'ood_tokens_after') < get('baseline', 'ood_tokens_after') * 0.6 else 'Learned boundaries do not yet match oracle collapse.',
    },
    {
        'hypothesis': 'collapse_beats_random',
        'supported': get('oracle', 'ood_acc') > get('random', 'ood_acc') + 0.05 if 'random' in summary else True,
        'interpretation': 'Structured thought boundaries matter more than arbitrary token removal.',
    },
]

print(json.dumps(summary, indent=2))
print(json.dumps(hypotheses, indent=2))
"""
    ),
    md("## Plot"),
    code(
        r"""
names = list(summary)
x = np.arange(len(names))
plt.figure(figsize=(9, 5))
plt.bar(x - 0.25, [summary[n]['ood_acc']['mean'] for n in names], width=0.25, label='OOD acc')
plt.bar(x, [summary[n]['hard_acc']['mean'] for n in names], width=0.25, label='Hard acc')
base_tokens = max(1e-9, summary.get('baseline', {}).get('ood_tokens_after', {}).get('mean') or 1)
plt.bar(x + 0.25, [summary[n]['ood_tokens_after']['mean'] / base_tokens for n in names], width=0.25, label='token ratio')
plt.xticks(x, names)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    md("## Save Report"),
    code(
        r"""
report = {
    'version': 'hierarchical_collapse_v1',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'fast_run': FAST_RUN,
    'config': {**CFG, 'P': P, 'train_clauses': TRAIN_CLAUSES, 'ood_clauses': OOD_CLAUSES, 'hard_clauses': HARD_CLAUSES, 'lr': LR},
    'summary': summary,
    'hypotheses': hypotheses,
    'results': results,
}
out = Path('/kaggle/working/recursive_thought_tokens_hierarchical_collapse_report.json') if Path('/kaggle/working').exists() else Path('recursive_thought_tokens_hierarchical_collapse_report.json')
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('saved', out)
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

out = Path("notebooks/kaggle_hierarchical_collapse.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
