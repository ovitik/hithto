"""Build a Kaggle notebook for learned reusable thought-token vocabularies."""

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
# Learned Higher-Level Thought Tokens

This notebook tests the closer version of the original idea:

> Higher-level thought tokens are learned during training, then reused during inference as ready-made abstractions.

This is different from repeatedly building thoughts at inference time. Here we learn a reusable **thought vocabulary**:

- primitive operation tokens describe small transformations;
- chunk tokenizers learn discrete macro-thought tokens for 2-step and 4-step chunks;
- inference models then reason over fewer, higher-level tokens.

We compare:

1. primitive baseline: all primitive tokens at inference;
2. learned level-1 thought tokens: one token replaces 2 primitive operations;
3. learned level-2 thought tokens: one token replaces 4 primitive operations;
4. oracle macro tokens, to separate tokenizer errors from reasoner limits.

Metrics: accuracy, OOD depth extrapolation, hard-depth extrapolation, tokens per example, and throughput.
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
    CFG = dict(steps_tokenizer=120, steps_reasoner=120, batch_size=128, eval_batches=3, seeds=[1], d_model=96)
else:
    CFG = dict(steps_tokenizer=1200, steps_reasoner=1600, batch_size=512, eval_batches=20, seeds=[1, 2, 3], d_model=128)

P = 31
N_SIGNS = 2
N_OPS = N_SIGNS * P
N_MACROS = N_OPS
TRAIN_LEN = (4, 8)
OOD_LEN = (12, 20)
HARD_LEN = (24, 32)
LR = 3e-4
TOKENIZER_LR = 5e-2

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
print(json.dumps(CFG, indent=2))
"""
    ),
    md(
        r"""
## Synthetic Algebra

Each primitive token is an affine transformation over modulo `P`:

```text
x + b
-x + b
```

A sequence of primitive operations composes into one macro operation of the same form. Therefore a chunk of 2 or 4 primitive tokens has a reusable higher-level meaning: a learned macro-thought token.

This is intentionally not natural language. It is a clean proof test for reusable compositional thought tokens.
"""
    ),
    code(
        r"""
def decode_op(op):
    sign_bit = op // P
    b = op % P
    s = torch.where(sign_bit == 0, torch.ones_like(b), -torch.ones_like(b))
    return s, b


def encode_op(s, b):
    sign_bit = (s < 0).long()
    return sign_bit * P + (b % P)


def compose_effect(e1, e2):
    # Apply e1 first, then e2. If e(x)=s*x+b, e2(e1(x)) = s2*s1*x + s2*b1+b2.
    s1, b1 = decode_op(e1)
    s2, b2 = decode_op(e2)
    s = s2 * s1
    b = (s2 * b1 + b2) % P
    return encode_op(s, b)


def compose_sequence(ops):
    effect = torch.zeros(ops.shape[:-1], dtype=torch.long, device=ops.device)
    for i in range(ops.shape[-1]):
        effect = compose_effect(effect, ops[..., i])
    return effect


def apply_effect(x, effect):
    s, b = decode_op(effect)
    return (s * x + b) % P


def make_ops(batch, length):
    return torch.randint(0, N_OPS, (batch, length), dtype=torch.long, device=device)


def make_chunk_batch(batch, chunk_size):
    ops = make_ops(batch, chunk_size)
    target = compose_sequence(ops)
    return ops, target


@dataclass
class ReasonBatch:
    start: torch.Tensor
    ops: torch.Tensor
    answer: torch.Tensor
    length: int


def make_reason_batch(batch, min_len, max_len):
    length = random.randint(min_len, max_len)
    start = torch.randint(0, P, (batch,), dtype=torch.long, device=device)
    ops = make_ops(batch, length)
    effect = compose_sequence(ops)
    answer = apply_effect(start, effect)
    return ReasonBatch(start=start, ops=ops, answer=answer, length=length)


def chunk_ops(ops, chunk_size):
    B, L = ops.shape
    pad = (-L) % chunk_size
    if pad:
        identity = torch.zeros((B, pad), dtype=torch.long, device=ops.device)
        ops = torch.cat([ops, identity], dim=1)
    return ops.view(B, -1, chunk_size)


print('example chunk target:', make_chunk_batch(2, 4))
"""
    ),
    md("## Models"),
    code(
        r"""
class ChunkTokenizer(nn.Module):
    def __init__(self, chunk_size, vocab_size=N_OPS, d_model=CFG['d_model']):
        super().__init__()
        if chunk_size != 2:
            raise ValueError('This notebook learns hierarchy by pairwise composition only.')
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.table = nn.Embedding(vocab_size * vocab_size, N_MACROS)

    def forward(self, ops):
        pair_id = ops[:, 0] * self.vocab_size + ops[:, 1]
        return self.table(pair_id)

    @torch.no_grad()
    def encode(self, ops):
        shape = ops.shape[:-1]
        flat = ops.reshape(-1, ops.shape[-1])
        return self(flat).argmax(-1).reshape(shape)


class Reasoner(nn.Module):
    def __init__(self, vocab_size, max_tokens=40, d_model=CFG['d_model'], n_layers=3, n_heads=4):
        super().__init__()
        self.entity = nn.Embedding(P, d_model)
        self.token = nn.Embedding(vocab_size, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1 + max_tokens, d_model) * 0.02)
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
        self.head = nn.Linear(d_model, P)

    def forward(self, start, tokens):
        B, T = tokens.shape
        q = self.entity(start)[:, None, :]
        x = torch.cat([q, self.token(tokens)], dim=1)
        types = torch.cat([
            torch.zeros((B, 1), dtype=torch.long, device=tokens.device),
            torch.ones((B, T), dtype=torch.long, device=tokens.device),
        ], dim=1)
        x = x + self.type_embed(types) + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(x)[:, 0])


class EffectReasoner(nn.Module):
    def __init__(self, max_tokens=40, d_model=CFG['d_model'], n_layers=2, n_heads=4):
        super().__init__()
        self.entity = nn.Embedding(P, d_model)
        self.sign_embed = nn.Embedding(2, d_model)
        self.offset_embed = nn.Embedding(P, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1 + max_tokens, d_model) * 0.02)
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
        self.head = nn.Linear(d_model, P)

    def forward(self, start, effects):
        B, T = effects.shape
        s, b = decode_op(effects)
        sign_bit = (s < 0).long()
        token = self.sign_embed(sign_bit) + self.offset_embed(b)
        q = self.entity(start)[:, None, :]
        x = torch.cat([q, token], dim=1)
        types = torch.cat([
            torch.zeros((B, 1), dtype=torch.long, device=effects.device),
            torch.ones((B, T), dtype=torch.long, device=effects.device),
        ], dim=1)
        x = x + self.type_embed(types) + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(x)[:, 0])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
"""
    ),
    md("## Tokenizer Training"),
    code(
        r"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tokenizer2(seed=1):
    set_seed(seed)
    chunk_size = 2
    model = ChunkTokenizer(chunk_size, vocab_size=N_OPS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TOKENIZER_LR, weight_decay=0.0)
    hist = []
    t0 = time.time()
    for step in range(1, CFG['steps_tokenizer'] + 1):
        model.train()
        ops, target = make_chunk_batch(CFG['batch_size'], chunk_size)
        logits = model(ops)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, CFG['steps_tokenizer'] // 4) == 0 or step == CFG['steps_tokenizer']:
            row = eval_tokenizer2(model, batches=max(1, CFG['eval_batches'] // 4))
            row.update({'step': step, 'loss': float(loss.detach().cpu()), 'elapsed_s': time.time() - t0})
            hist.append(row)
            print('tokenizer2', row)
    return model, hist


@torch.no_grad()
def eval_tokenizer2(model, batches=None):
    model.eval()
    batches = batches or CFG['eval_batches']
    correct = total = 0
    for _ in range(batches):
        ops, target = make_chunk_batch(CFG['batch_size'], 2)
        pred = model(ops).argmax(-1)
        correct += (pred == target).sum().item()
        total += target.numel()
    return {'acc': correct / total}


def make_macro_pair_batch(batch):
    left = torch.randint(0, N_MACROS, (batch,), dtype=torch.long, device=device)
    right = torch.randint(0, N_MACROS, (batch,), dtype=torch.long, device=device)
    pair = torch.stack([left, right], dim=1)
    target = compose_effect(left, right)
    return pair, target


def train_tokenizer4(seed=1):
    set_seed(seed)
    model = ChunkTokenizer(2, vocab_size=N_MACROS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TOKENIZER_LR, weight_decay=0.0)
    hist = []
    t0 = time.time()
    for step in range(1, CFG['steps_tokenizer'] + 1):
        model.train()
        pair, target = make_macro_pair_batch(CFG['batch_size'])
        logits = model(pair)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, CFG['steps_tokenizer'] // 4) == 0 or step == CFG['steps_tokenizer']:
            row = eval_tokenizer4(model, batches=max(1, CFG['eval_batches'] // 4))
            row.update({'step': step, 'loss': float(loss.detach().cpu()), 'elapsed_s': time.time() - t0})
            hist.append(row)
            print('tokenizer4', row)
    return model, hist


@torch.no_grad()
def eval_tokenizer4(model, batches=None):
    model.eval()
    batches = batches or CFG['eval_batches']
    correct = total = 0
    for _ in range(batches):
        pair, target = make_macro_pair_batch(CFG['batch_size'])
        pred = model(pair).argmax(-1)
        correct += (pred == target).sum().item()
        total += target.numel()
    return {'acc': correct / total}


tok2, tok2_hist = train_tokenizer2(seed=1)
tok4, tok4_hist = train_tokenizer4(seed=1)
print('tokenizer2 final', eval_tokenizer2(tok2))
print('tokenizer4 final', eval_tokenizer4(tok4))
"""
    ),
    md("## Reasoner Utilities"),
    code(
        r"""
@torch.no_grad()
def represent(batch, mode, learned=True):
    if mode == 'primitive':
        return batch.ops
    if mode == 'macro2':
        chunks = chunk_ops(batch.ops, 2)
        if learned:
            return tok2.encode(chunks)
        return compose_sequence(chunks)
    if mode == 'macro4':
        chunks = chunk_ops(batch.ops, 4)
        if learned:
            left = tok2.encode(chunks[:, :, :2])
            right = tok2.encode(chunks[:, :, 2:])
            return tok4.encode(torch.stack([left, right], dim=-1))
        return compose_sequence(chunks)
    raise ValueError(mode)


def train_reasoner(mode, learned=True, seed=1, reasoner_type='id'):
    set_seed(seed)
    vocab_size = N_OPS if mode == 'primitive' else N_MACROS
    max_tokens = HARD_LEN[1] if mode == 'primitive' else math.ceil(HARD_LEN[1] / (2 if mode == 'macro2' else 4))
    model = EffectReasoner(max_tokens=max_tokens).to(device) if reasoner_type == 'effect' else Reasoner(vocab_size=vocab_size, max_tokens=max_tokens).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    hist = []
    t0 = time.time()
    for step in range(1, CFG['steps_reasoner'] + 1):
        model.train()
        batch = make_reason_batch(CFG['batch_size'], *TRAIN_LEN)
        tokens = represent(batch, mode, learned=learned)
        logits = model(batch.start, tokens)
        loss = F.cross_entropy(logits, batch.answer)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, CFG['steps_reasoner'] // 4) == 0 or step == CFG['steps_reasoner']:
            row = {
                'step': step,
                'loss': float(loss.detach().cpu()),
                'train_acc': (logits.argmax(-1) == batch.answer).float().mean().item(),
                'ood_acc': eval_reasoner(model, mode, *OOD_LEN, learned=learned, batches=max(1, CFG['eval_batches'] // 4))['acc'],
                'elapsed_s': time.time() - t0,
            }
            hist.append(row)
            print(f'reasoner_{mode}_{reasoner_type}_learned{learned}', row)
    return model, hist


@torch.no_grad()
def eval_reasoner(model, mode, min_len, max_len, learned=True, batches=None):
    model.eval()
    batches = batches or CFG['eval_batches']
    correct = total = loss_sum = token_total = 0
    by_len = {l: [0, 0] for l in range(min_len, max_len + 1)}
    for _ in range(batches):
        batch = make_reason_batch(CFG['batch_size'], min_len, max_len)
        tokens = represent(batch, mode, learned=learned)
        logits = model(batch.start, tokens)
        pred = logits.argmax(-1)
        loss_sum += F.cross_entropy(logits, batch.answer, reduction='sum').item()
        correct += (pred == batch.answer).sum().item()
        total += batch.answer.numel()
        token_total += tokens.numel()
        by_len[batch.length][0] += (pred == batch.answer).sum().item()
        by_len[batch.length][1] += batch.answer.numel()
    return {
        'acc': correct / total,
        'loss': loss_sum / total,
        'avg_tokens': token_total / total,
        'by_len': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_len.items()},
    }


@torch.no_grad()
def benchmark(model, mode, learned=True, batches=30):
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    total = 0
    for _ in range(batches):
        batch = make_reason_batch(CFG['batch_size'], *OOD_LEN)
        tokens = represent(batch, mode, learned=learned)
        _ = model(batch.start, tokens)
        total += batch.answer.numel()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {'examples_per_s': total / elapsed, 'elapsed_s': elapsed}


@torch.no_grad()
def eval_symbolic_macro(mode, min_len, max_len, learned=False, batches=None):
    batches = batches or CFG['eval_batches']
    correct = total = token_total = 0
    for _ in range(batches):
        batch = make_reason_batch(CFG['batch_size'], min_len, max_len)
        tokens = represent(batch, mode, learned=learned)
        effect = compose_sequence(tokens)
        pred = apply_effect(batch.start, effect)
        correct += (pred == batch.answer).sum().item()
        total += batch.answer.numel()
        token_total += tokens.numel()
    return {'acc': correct / total, 'avg_tokens': token_total / total}
"""
    ),
    md("## Run Reasoner Experiments"),
    code(
        r"""
EXPERIMENTS = [
    dict(name='primitive_baseline', mode='primitive', learned=True),
    dict(name='macro2_learned', mode='macro2', learned=True),
    dict(name='macro2_oracle', mode='macro2', learned=False),
    dict(name='macro4_learned', mode='macro4', learned=True),
    dict(name='macro4_oracle', mode='macro4', learned=False),
    dict(name='macro2_effect_oracle', mode='macro2', learned=False, reasoner_type='effect'),
    dict(name='macro4_effect_oracle', mode='macro4', learned=False, reasoner_type='effect'),
]

if FAST_RUN:
    EXPERIMENTS = EXPERIMENTS[:3]

results = {}
for exp in EXPERIMENTS:
    seeds = CFG['seeds'] if exp['name'] in {'primitive_baseline', 'macro2_learned', 'macro4_learned'} else [CFG['seeds'][0]]
    for seed in seeds:
        run = f"{exp['name']}_seed{seed}"
        print('\\n===', run, '===')
        model, hist = train_reasoner(exp['mode'], learned=exp['learned'], seed=seed, reasoner_type=exp.get('reasoner_type', 'id'))
        row = {
            'spec': exp,
            'seed': seed,
            'params_reasoner': count_params(model),
            'history': hist,
            'iid': eval_reasoner(model, exp['mode'], *TRAIN_LEN, learned=exp['learned']),
            'ood': eval_reasoner(model, exp['mode'], *OOD_LEN, learned=exp['learned']),
            'hard': eval_reasoner(model, exp['mode'], *HARD_LEN, learned=exp['learned'], batches=max(4, CFG['eval_batches'] // 2)),
            'benchmark': benchmark(model, exp['mode'], learned=exp['learned'], batches=5 if FAST_RUN else 25),
        }
        results[run] = row
        print(json.dumps({k: row[k] for k in ['params_reasoner', 'iid', 'ood', 'hard', 'benchmark']}, indent=2))

symbolic_results = {
    'macro2_symbolic_oracle': {
        'iid': eval_symbolic_macro('macro2', *TRAIN_LEN, learned=False),
        'ood': eval_symbolic_macro('macro2', *OOD_LEN, learned=False),
        'hard': eval_symbolic_macro('macro2', *HARD_LEN, learned=False),
    },
    'macro4_symbolic_oracle': {
        'iid': eval_symbolic_macro('macro4', *TRAIN_LEN, learned=False),
        'ood': eval_symbolic_macro('macro4', *OOD_LEN, learned=False),
        'hard': eval_symbolic_macro('macro4', *HARD_LEN, learned=False),
    },
    'macro2_symbolic_learned': {
        'iid': eval_symbolic_macro('macro2', *TRAIN_LEN, learned=True),
        'ood': eval_symbolic_macro('macro2', *OOD_LEN, learned=True),
        'hard': eval_symbolic_macro('macro2', *HARD_LEN, learned=True),
    },
    'macro4_symbolic_learned': {
        'iid': eval_symbolic_macro('macro4', *TRAIN_LEN, learned=True),
        'ood': eval_symbolic_macro('macro4', *OOD_LEN, learned=True),
        'hard': eval_symbolic_macro('macro4', *HARD_LEN, learned=True),
    },
}
print('symbolic', json.dumps(symbolic_results, indent=2))
"""
    ),
    md("## Summary And Verdicts"),
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
        'params_reasoner': mean_std([r['params_reasoner'] for r in rows]),
        'iid_acc': mean_std([r['iid']['acc'] for r in rows]),
        'ood_acc': mean_std([r['ood']['acc'] for r in rows]),
        'hard_acc': mean_std([r['hard']['acc'] for r in rows]),
        'ood_avg_tokens': mean_std([r['ood']['avg_tokens'] for r in rows]),
        'hard_avg_tokens': mean_std([r['hard']['avg_tokens'] for r in rows]),
        'examples_per_s': mean_std([r['benchmark']['examples_per_s'] for r in rows]),
    }

tokenizer_summary = {
    'tokenizer2': {'params': count_params(tok2), 'eval': eval_tokenizer2(tok2), 'history': tok2_hist},
    'tokenizer4': {'params': count_params(tok4), 'eval': eval_tokenizer4(tok4), 'history': tok4_hist},
}

def get(name, metric):
    return summary.get(name, {}).get(metric, {}).get('mean') or 0.0

hypotheses = [
    {
        'hypothesis': 'learned_macro2_beats_primitive',
        'supported': get('macro2_learned', 'ood_acc') > get('primitive_baseline', 'ood_acc') + 0.03,
        'interpretation': 'Learned level-1 thought tokens improve OOD reasoning over primitive tokens.' if get('macro2_learned', 'ood_acc') > get('primitive_baseline', 'ood_acc') + 0.03 else 'Learned level-1 thought tokens do not improve OOD accuracy over primitive tokens.',
    },
    {
        'hypothesis': 'macro_tokens_reduce_inference_tokens',
        'supported': get('macro4_learned', 'ood_avg_tokens') < get('primitive_baseline', 'ood_avg_tokens') * 0.4 if 'macro4_learned' in summary else get('macro2_learned', 'ood_avg_tokens') < get('primitive_baseline', 'ood_avg_tokens') * 0.6,
        'interpretation': 'Learned thought tokens substantially reduce inference token count.',
    },
    {
        'hypothesis': 'learned_close_to_oracle',
        'supported': abs(get('macro2_oracle', 'ood_acc') - get('macro2_learned', 'ood_acc')) < 0.05,
        'interpretation': 'Learned macro tokenizer is close to oracle macro tokens.' if abs(get('macro2_oracle', 'ood_acc') - get('macro2_learned', 'ood_acc')) < 0.05 else 'Tokenizer errors or learned-code mismatch still hurt performance.',
    },
    {
        'hypothesis': 'meaning_aware_consumer_needed',
        'supported': get('macro4_effect_oracle', 'ood_acc') > get('macro4_oracle', 'ood_acc') + 0.10 or symbolic_results['macro4_symbolic_oracle']['ood']['acc'] > get('macro4_oracle', 'ood_acc') + 0.10,
        'interpretation': 'Macro tokens need a consumer that can use their meaning, not just their discrete IDs.',
    },
]

print(json.dumps(tokenizer_summary, indent=2))
print(json.dumps(summary, indent=2))
print(json.dumps(hypotheses, indent=2))
"""
    ),
    md("## Plots"),
    code(
        r"""
names = list(summary)
x = np.arange(len(names))
plt.figure(figsize=(max(9, len(names) * 1.0), 5))
plt.bar(x - 0.25, [summary[n]['ood_acc']['mean'] for n in names], width=0.25, label='OOD acc')
plt.bar(x, [summary[n]['hard_acc']['mean'] for n in names], width=0.25, label='Hard acc')
plt.bar(x + 0.25, [summary[n]['ood_avg_tokens']['mean'] / max(1, get('primitive_baseline', 'ood_avg_tokens')) for n in names], width=0.25, label='token ratio')
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
    'version': 'learned_thought_vocabulary_v1',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'fast_run': FAST_RUN,
    'config': {**CFG, 'P': P, 'n_ops': N_OPS, 'train_len': TRAIN_LEN, 'ood_len': OOD_LEN, 'hard_len': HARD_LEN, 'lr': LR, 'tokenizer_lr': TOKENIZER_LR},
    'tokenizer_summary': tokenizer_summary,
    'summary': summary,
    'symbolic_results': symbolic_results,
    'hypotheses': hypotheses,
    'results': results,
}
out = Path('/kaggle/working/recursive_thought_tokens_learned_vocabulary_report.json') if Path('/kaggle/working').exists() else Path('recursive_thought_tokens_learned_vocabulary_report.json')
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

out = Path("notebooks/kaggle_learned_thought_vocabulary.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
