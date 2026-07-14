"""Build a Kaggle notebook for meaning-aware neural composition of thought tokens."""

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
# Meaning-Aware Thought Token Composer

The previous notebook showed:

- learned higher-level thought tokens can be perfect and reusable;
- symbolic consumers use them perfectly;
- generic Transformer consumers do not learn their compositional meaning.

This notebook tests the next hypothesis:

> Higher-level thought tokens become useful reasoning units when the model has a meaning-aware neural composer.

We keep the same synthetic affine operators:

```text
effect token = s*x + b  mod P
```

But now the model is forced to represent each thought token as a continuous operator state `(sign, offset)` and learn neural composition:

```text
state <- compose(state, thought_token)
answer <- apply(state, start)
```

This is not symbolic hardcoding: the composition and application are trained neural modules. The symbolic path remains only an upper-bound diagnostic.
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
    CFG = dict(steps=200, batch_size=128, eval_batches=4, seeds=[1], d_model=96)
else:
    CFG = dict(steps=2400, batch_size=512, eval_batches=24, seeds=[1, 2, 3], d_model=128)

P = 31
N_OPS = 2 * P
TRAIN_LEN = (4, 8)
OOD_LEN = (12, 20)
HARD_LEN = (24, 32)
LR = 3e-4

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
print(json.dumps(CFG, indent=2))
"""
    ),
    md("## Algebra And Data"),
    code(
        r"""
def decode_op(op):
    sign_bit = op // P
    b = op % P
    s = torch.where(sign_bit == 0, torch.ones_like(b), -torch.ones_like(b))
    return s, b


def encode_op(s, b):
    return (s < 0).long() * P + (b % P)


def compose_effect(e1, e2):
    s1, b1 = decode_op(e1)
    s2, b2 = decode_op(e2)
    return encode_op(s2 * s1, (s2 * b1 + b2) % P)


def compose_sequence(ops):
    effect = torch.zeros(ops.shape[:-1], dtype=torch.long, device=ops.device)
    for i in range(ops.shape[-1]):
        effect = compose_effect(effect, ops[..., i])
    return effect


def apply_effect(x, effect):
    s, b = decode_op(effect)
    return (s * x + b) % P


@dataclass
class Batch:
    start: torch.Tensor
    ops: torch.Tensor
    answer: torch.Tensor
    effect: torch.Tensor
    length: int


def make_batch(batch_size, min_len, max_len):
    length = random.randint(min_len, max_len)
    start = torch.randint(0, P, (batch_size,), dtype=torch.long, device=device)
    ops = torch.randint(0, N_OPS, (batch_size, length), dtype=torch.long, device=device)
    effect = compose_sequence(ops)
    answer = apply_effect(start, effect)
    return Batch(start=start, ops=ops, answer=answer, effect=effect, length=length)


def chunk_ops(ops, chunk_size):
    B, L = ops.shape
    pad = (-L) % chunk_size
    if pad:
        ops = torch.cat([ops, torch.zeros(B, pad, dtype=torch.long, device=ops.device)], dim=1)
    return ops.view(B, -1, chunk_size)


def macro_tokens(ops, chunk_size):
    chunks = chunk_ops(ops, chunk_size)
    return compose_sequence(chunks)


print(make_batch(2, 4, 6))
"""
    ),
    md("## Models"),
    code(
        r"""
class GenericTransformerReasoner(nn.Module):
    def __init__(self, vocab_size=N_OPS, max_tokens=40, d_model=CFG['d_model'], n_layers=3, n_heads=4):
        super().__init__()
        self.entity = nn.Embedding(P, d_model)
        self.token = nn.Embedding(vocab_size, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1 + max_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=0.05, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, P)

    def forward(self, start, tokens):
        B, T = tokens.shape
        x = torch.cat([self.entity(start)[:, None], self.token(tokens)], dim=1)
        types = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=tokens.device), torch.ones(B, T, dtype=torch.long, device=tokens.device)], dim=1)
        x = x + self.type_embed(types) + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(x)[:, 0])


class MeaningAwareComposer(nn.Module):
    def __init__(self, d_model=CFG['d_model'], hidden_mult=2):
        super().__init__()
        self.sign_embed = nn.Embedding(2, d_model)
        self.offset_embed = nn.Embedding(P, d_model)
        self.identity = nn.Parameter(torch.randn(d_model) * 0.02)
        self.compose = nn.Sequential(
            nn.Linear(2 * d_model, hidden_mult * d_model),
            nn.GELU(),
            nn.Linear(hidden_mult * d_model, d_model),
        )
        self.effect_head = nn.Linear(d_model, N_OPS)
        self.start_embed = nn.Embedding(P, d_model)
        self.apply_net = nn.Sequential(
            nn.Linear(2 * d_model, hidden_mult * d_model),
            nn.GELU(),
            nn.Linear(hidden_mult * d_model, P),
        )

    def token_embed(self, tokens):
        s, b = decode_op(tokens)
        sign_bit = (s < 0).long()
        return self.sign_embed(sign_bit) + self.offset_embed(b)

    def forward(self, start, tokens, return_effect=False):
        B, T = tokens.shape
        state = self.identity[None, :].expand(B, -1)
        embeds = self.token_embed(tokens)
        states = []
        for i in range(T):
            state = self.compose(torch.cat([state, embeds[:, i]], dim=-1))
            states.append(state)
        logits = self.apply_net(torch.cat([self.start_embed(start), state], dim=-1))
        if return_effect:
            return logits, self.effect_head(state), torch.stack(states, dim=1)
        return logits


class SupervisedComposer(MeaningAwareComposer):
    pass


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
"""
    ),
    md("## Representation And Evaluation"),
    code(
        r"""
def make_tokens(batch, mode):
    if mode == 'primitive':
        return batch.ops
    if mode == 'macro2':
        return macro_tokens(batch.ops, 2)
    if mode == 'macro4':
        return macro_tokens(batch.ops, 4)
    raise ValueError(mode)


@torch.no_grad()
def eval_model(model, mode, min_len, max_len, batches=None, effect_aux=False):
    model.eval()
    batches = batches or CFG['eval_batches']
    correct = total = loss_sum = token_total = effect_correct = 0
    by_len = {l: [0, 0] for l in range(min_len, max_len + 1)}
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_len, max_len)
        tokens = make_tokens(batch, mode)
        if effect_aux:
            logits, eff_logits, _ = model(batch.start, tokens, return_effect=True)
            effect_correct += (eff_logits.argmax(-1) == batch.effect).sum().item()
        else:
            logits = model(batch.start, tokens)
        pred = logits.argmax(-1)
        loss_sum += F.cross_entropy(logits, batch.answer, reduction='sum').item()
        correct += (pred == batch.answer).sum().item()
        total += batch.answer.numel()
        token_total += tokens.numel()
        by_len[batch.length][0] += (pred == batch.answer).sum().item()
        by_len[batch.length][1] += batch.answer.numel()
    out = {
        'acc': correct / total,
        'loss': loss_sum / total,
        'avg_tokens': token_total / total,
        'by_len': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_len.items()},
    }
    if effect_aux:
        out['effect_acc'] = effect_correct / total
    return out


@torch.no_grad()
def eval_symbolic(mode, min_len, max_len, batches=None):
    batches = batches or CFG['eval_batches']
    correct = total = token_total = 0
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_len, max_len)
        tokens = make_tokens(batch, mode)
        effect = compose_sequence(tokens)
        pred = apply_effect(batch.start, effect)
        correct += (pred == batch.answer).sum().item()
        total += batch.answer.numel()
        token_total += tokens.numel()
    return {'acc': correct / total, 'avg_tokens': token_total / total}


@torch.no_grad()
def benchmark(model, mode, batches=30):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    total = 0
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], *OOD_LEN)
        _ = model(batch.start, make_tokens(batch, mode))
        total += batch.answer.numel()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {'examples_per_s': total / elapsed, 'elapsed_s': elapsed}
"""
    ),
    md("## Training"),
    code(
        r"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(kind, mode):
    max_tokens = HARD_LEN[1] if mode == 'primitive' else math.ceil(HARD_LEN[1] / (2 if mode == 'macro2' else 4))
    if kind == 'generic':
        return GenericTransformerReasoner(vocab_size=N_OPS, max_tokens=max_tokens)
    if kind in ('composer', 'composer_effect_aux'):
        return MeaningAwareComposer()
    raise ValueError(kind)


def train_model(kind, mode, seed=1):
    set_seed(seed)
    model = build_model(kind, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    hist = []
    t0 = time.time()
    effect_aux = kind == 'composer_effect_aux'
    for step in range(1, CFG['steps'] + 1):
        model.train()
        batch = make_batch(CFG['batch_size'], *TRAIN_LEN)
        tokens = make_tokens(batch, mode)
        if effect_aux:
            logits, eff_logits, _ = model(batch.start, tokens, return_effect=True)
            answer_loss = F.cross_entropy(logits, batch.answer)
            effect_loss = F.cross_entropy(eff_logits, batch.effect)
            loss = answer_loss + 0.5 * effect_loss
        else:
            logits = model(batch.start, tokens)
            answer_loss = F.cross_entropy(logits, batch.answer)
            effect_loss = torch.tensor(0.0, device=device)
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
                'effect_loss': float(effect_loss.detach().cpu()),
                'train_acc': (logits.argmax(-1) == batch.answer).float().mean().item(),
                'ood_acc': eval_model(model, mode, *OOD_LEN, batches=max(1, CFG['eval_batches'] // 4), effect_aux=effect_aux)['acc'],
                'elapsed_s': time.time() - t0,
            }
            hist.append(row)
            print(f'{kind}_{mode}', row)
    return model, hist
"""
    ),
    md("## Run Experiments"),
    code(
        r"""
EXPERIMENTS = [
    dict(name='generic_primitive', kind='generic', mode='primitive'),
    dict(name='generic_macro4', kind='generic', mode='macro4'),
    dict(name='composer_primitive', kind='composer', mode='primitive'),
    dict(name='composer_macro2', kind='composer', mode='macro2'),
    dict(name='composer_macro4', kind='composer', mode='macro4'),
    dict(name='composer_aux_macro4', kind='composer_effect_aux', mode='macro4'),
]

if FAST_RUN:
    EXPERIMENTS = EXPERIMENTS[:4]

results = {}
for exp in EXPERIMENTS:
    seeds = CFG['seeds'] if exp['name'] in {'generic_primitive', 'composer_macro4', 'composer_aux_macro4'} else [CFG['seeds'][0]]
    for seed in seeds:
        run = f"{exp['name']}_seed{seed}"
        print('\\n===', run, '===')
        model, hist = train_model(exp['kind'], exp['mode'], seed=seed)
        effect_aux = exp['kind'] == 'composer_effect_aux'
        row = {
            'spec': exp,
            'seed': seed,
            'params': count_params(model),
            'history': hist,
            'iid': eval_model(model, exp['mode'], *TRAIN_LEN, effect_aux=effect_aux),
            'ood': eval_model(model, exp['mode'], *OOD_LEN, effect_aux=effect_aux),
            'hard': eval_model(model, exp['mode'], *HARD_LEN, batches=max(4, CFG['eval_batches'] // 2), effect_aux=effect_aux),
            'benchmark': benchmark(model, exp['mode'], batches=5 if FAST_RUN else 25),
        }
        results[run] = row
        print(json.dumps({k: row[k] for k in ['params', 'iid', 'ood', 'hard', 'benchmark']}, indent=2))

symbolic = {
    mode: {
        'iid': eval_symbolic(mode, *TRAIN_LEN),
        'ood': eval_symbolic(mode, *OOD_LEN),
        'hard': eval_symbolic(mode, *HARD_LEN),
    }
    for mode in ['primitive', 'macro2', 'macro4']
}
print('symbolic', json.dumps(symbolic, indent=2))
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
    groups.setdefault(row['spec']['name'], []).append(row)

summary = {}
for name, rows in groups.items():
    summary[name] = {
        'params': mean_std([r['params'] for r in rows]),
        'iid_acc': mean_std([r['iid']['acc'] for r in rows]),
        'ood_acc': mean_std([r['ood']['acc'] for r in rows]),
        'hard_acc': mean_std([r['hard']['acc'] for r in rows]),
        'ood_avg_tokens': mean_std([r['ood']['avg_tokens'] for r in rows]),
        'hard_avg_tokens': mean_std([r['hard']['avg_tokens'] for r in rows]),
        'ood_effect_acc': mean_std([r['ood'].get('effect_acc') for r in rows if 'effect_acc' in r['ood']]),
        'examples_per_s': mean_std([r['benchmark']['examples_per_s'] for r in rows]),
    }

def get(name, metric):
    return summary.get(name, {}).get(metric, {}).get('mean') or 0.0

hypotheses = [
    {
        'hypothesis': 'meaning_aware_composer_beats_generic',
        'supported': get('composer_macro4', 'ood_acc') > get('generic_macro4', 'ood_acc') + 0.10,
        'interpretation': 'A meaning-aware composer uses macro thought tokens better than a generic Transformer.' if get('composer_macro4', 'ood_acc') > get('generic_macro4', 'ood_acc') + 0.10 else 'Meaning-aware composer does not yet beat generic Transformer by a large margin.',
    },
    {
        'hypothesis': 'macro4_reduces_tokens_without_losing_accuracy',
        'supported': get('composer_macro4', 'ood_avg_tokens') < get('composer_primitive', 'ood_avg_tokens') * 0.4 and get('composer_macro4', 'ood_acc') >= get('composer_primitive', 'ood_acc') - 0.05,
        'interpretation': 'Macro thought tokens reduce inference length while preserving composer accuracy.',
    },
    {
        'hypothesis': 'effect_aux_helps_composition',
        'supported': get('composer_aux_macro4', 'ood_acc') > get('composer_macro4', 'ood_acc') + 0.03,
        'interpretation': 'Auxiliary supervision of composed effect improves macro-token reasoning.' if get('composer_aux_macro4', 'ood_acc') > get('composer_macro4', 'ood_acc') + 0.03 else 'Auxiliary effect supervision does not materially improve macro-token reasoning.',
    },
]

print(json.dumps(summary, indent=2))
print(json.dumps(symbolic, indent=2))
print(json.dumps(hypotheses, indent=2))
"""
    ),
    md("## Plot"),
    code(
        r"""
names = list(summary)
x = np.arange(len(names))
plt.figure(figsize=(max(10, len(names) * 0.9), 5))
plt.bar(x - 0.25, [summary[n]['ood_acc']['mean'] for n in names], width=0.25, label='OOD acc')
plt.bar(x, [summary[n]['hard_acc']['mean'] for n in names], width=0.25, label='Hard acc')
base_tokens = max(1e-9, summary.get('generic_primitive', {}).get('ood_avg_tokens', {}).get('mean') or 1)
plt.bar(x + 0.25, [summary[n]['ood_avg_tokens']['mean'] / base_tokens for n in names], width=0.25, label='token ratio')
plt.xticks(x, names, rotation=35, ha='right')
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
    'version': 'meaning_aware_composer_v1',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'fast_run': FAST_RUN,
    'config': {**CFG, 'P': P, 'n_ops': N_OPS, 'train_len': TRAIN_LEN, 'ood_len': OOD_LEN, 'hard_len': HARD_LEN, 'lr': LR},
    'summary': summary,
    'symbolic': symbolic,
    'hypotheses': hypotheses,
    'results': results,
}
out = Path('/kaggle/working/recursive_thought_tokens_meaning_aware_composer_report.json') if Path('/kaggle/working').exists() else Path('recursive_thought_tokens_meaning_aware_composer_report.json')
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

out = Path("notebooks/kaggle_meaning_aware_composer.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
