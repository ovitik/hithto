"""Build a focused Kaggle notebook for branch-trap and learned-halting tests."""

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
# Recursive Thought Tokens: Focused Halting + Branch-Trap Test

This notebook is a targeted follow-up to the v3 report. It does not try to retest everything.

It focuses only on the two remaining weak spots:

1. **Learned halting**: can the model choose the right recursive step without oracle-depth evaluation?
2. **Branch-trap reasoning**: can it follow the correct branch when several plausible edges compete?

This should run faster than the ultimate notebook because it removes broad baselines, self-loop diagnostics, and unrelated task variants.
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

if FAST_RUN:
    CFG = dict(train_steps=80, batch_size=96, eval_batches=3, seeds=[1], d_model=96, n_entities=64, max_steps=8, distractors=8)
else:
    CFG = dict(train_steps=1800, batch_size=256, eval_batches=16, seeds=[1, 2, 3], d_model=128, n_entities=96, max_steps=10, distractors=16)

LR = 3e-4
N_RELATIONS = 4
TRAIN_DEPTH = (1, 3)
OOD_DEPTH = (4, CFG['max_steps'])
HARD_DEPTH = (CFG['max_steps'] + 1, CFG['max_steps'] + 4)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print('device:', device)
if device.type == 'cuda':
    print('gpu:', torch.cuda.get_device_name(0))
print(json.dumps(CFG, indent=2))
"""
    ),
    md("## Data\n\n`chain` tests halting without branch ambiguity. `branch_trap` adds wrong edges from the same source/relation, so the model must disambiguate the correct branch."),
    code(
        r"""
@dataclass
class Batch:
    facts: torch.Tensor
    query: torch.Tensor
    answer: torch.Tensor
    depth: torch.Tensor
    path_pos: torch.Tensor
    path_nodes: torch.Tensor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(batch_size, min_depth, max_depth, *, task='chain', device=device):
    n_entities = CFG['n_entities']
    distractors = CFG['distractors']
    max_steps = max(CFG['max_steps'], max_depth)
    is_branch = task == 'branch_trap'
    facts_list, queries, answers, depths, pos_list, node_list = [], [], [], [], [], []
    n_facts = max_depth + distractors + (max_depth if is_branch else 0)

    for _ in range(batch_size):
        depth = random.randint(min_depth, max_depth)
        start = random.randrange(n_entities)
        nodes = [start]
        used = {start}
        for _s in range(depth):
            nxt = random.randrange(n_entities)
            tries = 0
            while nxt in used and tries < 50:
                nxt = random.randrange(n_entities)
                tries += 1
            nodes.append(nxt)
            used.add(nxt)

        facts = []
        gold_old = []
        for step in range(depth):
            gold_old.append(len(facts))
            facts.append([nodes[step], 0, nodes[step + 1]])
            if is_branch:
                # Add a tempting wrong edge with the same source and relation.
                wrong = random.randrange(n_entities)
                while wrong == nodes[step + 1]:
                    wrong = random.randrange(n_entities)
                facts.append([nodes[step], 0, wrong])

        while len(facts) < n_facts:
            f = [random.randrange(n_entities), random.randrange(N_RELATIONS), random.randrange(n_entities)]
            if f not in facts:
                facts.append(f)

        perm = list(range(len(facts)))
        random.shuffle(perm)
        old_to_new = {old: new for new, old in enumerate(perm)}
        shuffled = [facts[i] for i in perm]
        path_pos = [old_to_new[i] for i in gold_old] + [-100] * max(0, max_steps - depth)
        path_nodes = nodes + [-100] * max(0, max_steps + 1 - len(nodes))

        facts_list.append(shuffled)
        queries.append(start)
        answers.append(nodes[-1])
        depths.append(depth)
        pos_list.append(path_pos[:max_steps])
        node_list.append(path_nodes[:max_steps + 1])

    return Batch(
        facts=torch.tensor(facts_list, dtype=torch.long, device=device),
        query=torch.tensor(queries, dtype=torch.long, device=device),
        answer=torch.tensor(answers, dtype=torch.long, device=device),
        depth=torch.tensor(depths, dtype=torch.long, device=device),
        path_pos=torch.tensor(pos_list, dtype=torch.long, device=device),
        path_nodes=torch.tensor(node_list, dtype=torch.long, device=device),
    )


batch = make_batch(1, 2, 3, task='branch_trap')
print('query', int(batch.query[0]), 'answer', int(batch.answer[0]), 'depth', int(batch.depth[0]))
print('facts', batch.facts[0].detach().cpu().tolist())
print('gold positions', batch.path_pos[0].detach().cpu().tolist())
"""
    ),
    md("## Models"),
    code(
        r"""
class EdgeDepthBaseline(nn.Module):
    def __init__(self, d_model=CFG['d_model'], n_layers=3, n_heads=4, max_facts=80):
        super().__init__()
        self.uses_depth = True
        self.entity = nn.Embedding(CFG['n_entities'], d_model)
        self.relation = nn.Embedding(N_RELATIONS, d_model)
        self.depth_embed = nn.Embedding(CFG['max_steps'] + 8, d_model)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.type_embed = nn.Embedding(3, d_model)
        self.pos = nn.Parameter(torch.randn(1, 2 + max_facts, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=0.05, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, CFG['n_entities'])

    def forward(self, facts, query, depth):
        B, F_, _ = facts.shape
        edge = self.edge_mlp(torch.cat([
            self.entity(facts[:, :, 0]),
            self.relation(facts[:, :, 1]),
            self.entity(facts[:, :, 2]),
        ], dim=-1))
        q = self.entity(query)[:, None, :]
        d = self.depth_embed(depth.clamp(0, self.depth_embed.num_embeddings - 1))[:, None, :]
        x = torch.cat([q, d, edge], dim=1)
        types = torch.cat([
            torch.zeros((B, 1), dtype=torch.long, device=facts.device),
            torch.full((B, 1), 2, dtype=torch.long, device=facts.device),
            torch.ones((B, F_), dtype=torch.long, device=facts.device),
        ], dim=1)
        x = x + self.type_embed(types) + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(x)[:, 0])


class RecursiveThoughtModel(nn.Module):
    def __init__(self, d_model=CFG['d_model'], max_steps=CFG['max_steps']):
        super().__init__()
        self.max_steps = max_steps
        self.entity = nn.Embedding(CFG['n_entities'], d_model)
        self.relation = nn.Embedding(N_RELATIONS, d_model)
        self.edge_mlp = nn.Sequential(nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.update = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, CFG['n_entities'], bias=False)
        self.node_head = nn.Linear(d_model, CFG['n_entities'])
        self.halt_head = nn.Linear(d_model, 1)
        self.out.weight = self.entity.weight

    def encode_edges(self, facts):
        return self.edge_mlp(torch.cat([
            self.entity(facts[:, :, 0]),
            self.relation(facts[:, :, 1]),
            self.entity(facts[:, :, 2]),
        ], dim=-1))

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
            ctx = torch.einsum('bf,bfd->bd', attn, vals)
            state = self.norm(self.update(ctx, state))
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('baseline params', count_params(EdgeDepthBaseline()))
print('thought params', count_params(RecursiveThoughtModel()))
"""
    ),
    md("## Train / Eval"),
    code(
        r"""
def step_logits(model, trace):
    return torch.einsum('bsd,nd->bsn', trace['states'][:, 1:], model.entity.weight)


def select_logits(model, batch, trace, mode):
    per_step = step_logits(model, trace)
    if mode == 'depth':
        idx = (batch.depth - 1).clamp(0, per_step.shape[1] - 1)
        return per_step[torch.arange(per_step.shape[0], device=device), idx]
    if mode == 'halt':
        w = torch.softmax(trace['halt_logits'], dim=-1)
        return torch.einsum('bs,bsn->bn', w, per_step)
    if mode == 'final':
        return per_step[:, -1]
    raise ValueError(mode)


def answer_mode_for_step(mode, step, steps):
    if mode == 'depth_then_halt':
        return 'depth' if step <= max(1, steps // 2) else 'halt'
    return mode


def train_depth_for_step(step, steps, schedule):
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


def aux_loss(model, batch, trace, mode, weight):
    if weight <= 0 or mode == 'none':
        return torch.tensor(0.0, device=device), {}
    losses, metrics = [], {}
    if 'attention' in mode:
        pos = batch.path_pos[:, :trace['attn'].shape[1]]
        mask = pos >= 0
        if mask.any():
            loss = F.nll_loss(trace['attn'].clamp_min(1e-8).log()[mask], pos[mask])
            losses.append(loss)
            metrics['attn_loss'] = float(loss.detach().cpu())
    if 'halt' in mode:
        target = (batch.depth - 1).clamp(0, trace['halt_logits'].shape[1] - 1)
        loss = F.cross_entropy(trace['halt_logits'], target)
        losses.append(loss)
        metrics['halt_loss'] = float(loss.detach().cpu())
    if not losses:
        return torch.tensor(0.0, device=device), metrics
    total = sum(losses) * weight
    metrics['aux_loss'] = float(total.detach().cpu())
    return total, metrics


@torch.no_grad()
def eval_model(model, min_depth, max_depth, *, task, answer_mode='depth', batches=None):
    model.eval()
    batches = batches or CFG['eval_batches']
    total, correct, loss_sum = 0, 0, 0.0
    oracle_depth_correct, halt_correct, any_correct = 0, 0, 0
    by_depth = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    by_depth_halt = {d: [0, 0] for d in range(min_depth, max_depth + 1)}
    for _ in range(batches):
        batch = make_batch(CFG['batch_size'], min_depth, max_depth, task=task)
        if isinstance(model, RecursiveThoughtModel):
            steps = max(max_depth, model.max_steps)
            _, trace = model(batch.facts, batch.query, steps=steps, return_trace=True)
            logits = select_logits(model, batch, trace, answer_mode)
            depth_logits = select_logits(model, batch, trace, 'depth')
            halt_logits = select_logits(model, batch, trace, 'halt')
            per_step_pred = step_logits(model, trace).argmax(-1)
            oracle_depth_correct += (depth_logits.argmax(-1) == batch.answer).sum().item()
            halt_correct += (halt_logits.argmax(-1) == batch.answer).sum().item()
            any_correct += (per_step_pred == batch.answer[:, None]).any(dim=1).sum().item()
            halt_pred = halt_logits.argmax(-1)
        else:
            logits = model(batch.facts, batch.query, batch.depth)
            halt_pred = None
        pred = logits.argmax(-1)
        loss_sum += F.cross_entropy(logits, batch.answer, reduction='sum').item()
        correct += (pred == batch.answer).sum().item()
        total += batch.answer.numel()
        for d in by_depth:
            mask = batch.depth == d
            if mask.any():
                by_depth[d][0] += (pred[mask] == batch.answer[mask]).sum().item()
                by_depth[d][1] += mask.sum().item()
                if halt_pred is not None:
                    by_depth_halt[d][0] += (halt_pred[mask] == batch.answer[mask]).sum().item()
                    by_depth_halt[d][1] += mask.sum().item()
    out = {
        'acc': correct / total,
        'loss': loss_sum / total,
        'by_depth': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth.items()},
    }
    if isinstance(model, RecursiveThoughtModel):
        out.update({
            'oracle_depth_acc': oracle_depth_correct / total,
            'halt_acc': halt_correct / total,
            'oracle_any_step_acc': any_correct / total,
            'by_depth_halt': {str(k): (v[0] / v[1] if v[1] else None) for k, v in by_depth_halt.items()},
        })
    return out


def train_model(model, name, *, task, answer_mode='depth', eval_answer_mode=None, aux_mode='attention', aux_weight=0.5, schedule='curriculum', steps=None, seed=1):
    set_seed(seed)
    model.to(device)
    steps = steps or CFG['train_steps']
    eval_answer_mode = eval_answer_mode or ('halt' if answer_mode == 'depth_then_halt' else answer_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    hist = []
    t0 = time.time()
    log_every = max(1, steps // 4)
    for step in range(1, steps + 1):
        model.train()
        depth_range = train_depth_for_step(step, steps, schedule)
        batch = make_batch(CFG['batch_size'], *depth_range, task=task)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            if isinstance(model, RecursiveThoughtModel):
                _, trace = model(batch.facts, batch.query, return_trace=True)
                step_mode = answer_mode_for_step(answer_mode, step, steps)
                logits = select_logits(model, batch, trace, step_mode)
                aux, extra = aux_loss(model, batch, trace, aux_mode, aux_weight)
            else:
                step_mode = 'final'
                logits = model(batch.facts, batch.query, batch.depth)
                aux, extra = torch.tensor(0.0, device=device), {}
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
                'loss': float(loss.detach().cpu()),
                'answer_loss': float(answer_loss.detach().cpu()),
                'train_acc': (logits.argmax(-1) == batch.answer).float().mean().item(),
                'step_mode': step_mode,
                'train_depth': depth_range,
                'elapsed_s': time.time() - t0,
            }
            row.update(extra)
            row['ood_acc'] = eval_model(model, *OOD_DEPTH, task=task, answer_mode=eval_answer_mode, batches=max(1, CFG['eval_batches'] // 4))['acc']
            hist.append(row)
            print(name, row)
    return hist
"""
    ),
    md("## Experiments\n\nThe matrix is intentionally small. It compares the strongest relevant baseline to recursive variants for the two focused questions."),
    code(
        r"""
EXPERIMENTS = [
    # Halting on clean chains.
    dict(name='baseline_depth_chain', model='baseline', task='chain', schedule='curriculum', answer_mode='final', eval_answer_mode='final', aux_mode='none'),
    dict(name='thought_depth_chain', model='thought', task='chain', schedule='curriculum', answer_mode='depth', eval_answer_mode='depth', aux_mode='attention', aux_weight=0.5),
    dict(name='thought_halt_chain', model='thought', task='chain', schedule='curriculum', answer_mode='halt', eval_answer_mode='halt', aux_mode='attention_halt', aux_weight=0.35),
    dict(name='thought_twostage_halt_chain', model='thought', task='chain', schedule='curriculum', answer_mode='depth_then_halt', eval_answer_mode='halt', aux_mode='attention_halt', aux_weight=0.35),

    # Branch-trap reasoning.
    dict(name='baseline_depth_branch_trap', model='baseline', task='branch_trap', schedule='curriculum', answer_mode='final', eval_answer_mode='final', aux_mode='none'),
    dict(name='thought_depth_branch_trap', model='thought', task='branch_trap', schedule='curriculum', answer_mode='depth', eval_answer_mode='depth', aux_mode='attention', aux_weight=0.5),
    dict(name='thought_halt_branch_trap', model='thought', task='branch_trap', schedule='curriculum', answer_mode='halt', eval_answer_mode='halt', aux_mode='attention_halt', aux_weight=0.35),
]

if FAST_RUN:
    EXPERIMENTS = EXPERIMENTS[:3]

def build_model(kind):
    return EdgeDepthBaseline() if kind == 'baseline' else RecursiveThoughtModel()

results = {}
global_t0 = time.time()
for exp in EXPERIMENTS:
    seeds = CFG['seeds'] if exp['name'] in {'baseline_depth_chain', 'thought_depth_chain', 'thought_halt_chain', 'thought_twostage_halt_chain'} else [CFG['seeds'][0]]
    for seed in seeds:
        run = f"{exp['name']}_seed{seed}"
        print('\\n===', run, '===')
        model = build_model(exp['model'])
        hist = train_model(
            model, run,
            task=exp['task'],
            answer_mode=exp['answer_mode'],
            eval_answer_mode=exp['eval_answer_mode'],
            aux_mode=exp.get('aux_mode', 'none'),
            aux_weight=exp.get('aux_weight', 0.5),
            schedule=exp['schedule'],
            seed=seed,
        )
        row = {
            'spec': exp,
            'seed': seed,
            'params': count_params(model),
            'history': hist,
            'iid': eval_model(model, *TRAIN_DEPTH, task=exp['task'], answer_mode=exp['eval_answer_mode']),
            'ood': eval_model(model, *OOD_DEPTH, task=exp['task'], answer_mode=exp['eval_answer_mode']),
            'hard_ood': eval_model(model, *HARD_DEPTH, task=exp['task'], answer_mode=exp['eval_answer_mode'], batches=max(4, CFG['eval_batches'] // 2)),
            'elapsed_s_total': time.time() - global_t0,
        }
        results[run] = row
        print(json.dumps({k: row[k] for k in ['params', 'iid', 'ood', 'hard_ood']}, indent=2))
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
    params = [r['params'] for r in rows]
    ood = [r['ood']['acc'] for r in rows]
    hard = [r['hard_ood']['acc'] for r in rows]
    summary[name] = {
        'params': mean_std(params),
        'iid_acc': mean_std([r['iid']['acc'] for r in rows]),
        'ood_acc': mean_std(ood),
        'hard_ood_acc': mean_std(hard),
        'ood_oracle_depth_acc': mean_std([r['ood'].get('oracle_depth_acc') for r in rows if 'oracle_depth_acc' in r['ood']]),
        'ood_halt_acc': mean_std([r['ood'].get('halt_acc') for r in rows if 'halt_acc' in r['ood']]),
        'ood_oracle_any_step_acc': mean_std([r['ood'].get('oracle_any_step_acc') for r in rows if 'oracle_any_step_acc' in r['ood']]),
        'ood_per_million_params': mean_std([a / (p / 1_000_000) for a, p in zip(ood, params)]),
        'hard_ood_per_million_params': mean_std([a / (p / 1_000_000) for a, p in zip(hard, params)]),
        'task': rows[0]['spec']['task'],
        'answer_mode': rows[0]['spec']['answer_mode'],
    }

def get(name, metric='ood_acc'):
    return summary.get(name, {}).get(metric, {}).get('mean') or 0.0

hypotheses = [
    {
        'hypothesis': 'halting_gap_on_chain',
        'supported': get('thought_halt_chain') >= 0.70 and get('thought_halt_chain') >= get('thought_depth_chain') - 0.08,
        'interpretation': 'Learned halting is close to oracle-depth on clean chains.' if get('thought_halt_chain') >= 0.70 and get('thought_halt_chain') >= get('thought_depth_chain') - 0.08 else 'Learned halting still trails oracle-depth or is not accurate enough on clean chains.',
    },
    {
        'hypothesis': 'twostage_halting_helps',
        'supported': get('thought_twostage_halt_chain') > get('thought_halt_chain') + 0.03,
        'interpretation': 'Two-stage depth-then-halt training improves learned halting.' if get('thought_twostage_halt_chain') > get('thought_halt_chain') + 0.03 else 'Two-stage depth-then-halt training does not improve learned halting.',
    },
    {
        'hypothesis': 'branch_trap_fixed',
        'supported': get('thought_depth_branch_trap') > get('baseline_depth_branch_trap') + 0.10 and get('thought_depth_branch_trap') > 0.50,
        'interpretation': 'Recursive thought solves branch-trap substantially better than the depth-aware baseline.' if get('thought_depth_branch_trap') > get('baseline_depth_branch_trap') + 0.10 and get('thought_depth_branch_trap') > 0.50 else 'Branch-trap remains an unresolved weakness.',
    },
]

print(json.dumps(summary, indent=2))
print(json.dumps(hypotheses, indent=2))
"""
    ),
    md("## Plots"),
    code(
        r"""
names = list(summary)
x = np.arange(len(names))
plt.figure(figsize=(max(9, len(names) * 0.8), 5))
plt.bar(x - 0.18, [summary[n]['ood_acc']['mean'] for n in names], width=0.36, label='OOD 4-max')
plt.bar(x + 0.18, [summary[n]['hard_ood_acc']['mean'] for n in names], width=0.36, label='Hard OOD')
plt.xticks(x, names, rotation=45, ha='right')
plt.ylim(0, 1.05)
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    md("## Save Report"),
    code(
        r"""
report = {
    'version': 'focused_branch_halting_v1',
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'device': str(device),
    'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'fast_run': FAST_RUN,
    'config': {**CFG, 'lr': LR, 'train_depth': TRAIN_DEPTH, 'ood_depth': OOD_DEPTH, 'hard_depth': HARD_DEPTH},
    'summary': summary,
    'hypotheses': hypotheses,
    'results': results,
}
out = Path('/kaggle/working/recursive_thought_tokens_focused_branch_halting_report.json') if Path('/kaggle/working').exists() else Path('recursive_thought_tokens_focused_branch_halting_report.json')
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

out = Path("notebooks/kaggle_focused_branch_halting.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
