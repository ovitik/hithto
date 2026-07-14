from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

PRIMITIVES = [f"<A{i:02d}>" for i in range(64)]
C2 = [f"<C2_{i:02d}>" for i in range(8)]
C3 = [f"<C3_{i:02d}>" for i in range(4)]
C4 = [f"<C4_{i:02d}>" for i in range(2)]
HA_START = "<HA_START>"
HA_ROOT = "<HA_ROOT>"
HA_END = "<HA_END>"
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
        return max((root.depth for root in self.roots), default=0)

    @property
    def node_count(self) -> int:
        return len(self.nodes_in_generation_order)


class HACoTGrammar:
    def __init__(self, max_roots: int = 8, max_nodes: int = 128, max_depth: int = 12) -> None:
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
                    raise ValueError("<HA_END> requires empty active stack")
                if not roots:
                    raise ValueError("<HA_END> requires at least one root")
                ended = True
            else:
                raise ValueError(f"unknown token: {tok}")
            if len(nodes) > self.max_nodes:
                raise ValueError("max semantic nodes exceeded")
        if not ended:
            raise ValueError("missing <HA_END>")
        return ParsedForest(tokens=tokens, roots=roots, nodes_in_generation_order=nodes)

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
        for root_i in range(roots):
            if remaining_nodes <= 0:
                break
            root_budget = max(1, remaining_nodes // max(1, roots - root_i))
            subtree, used = emit_subtree(rng.randint(1, self.max_depth), root_budget)
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


def flat_tokens_for_tree(tree_tokens: list[str]) -> list[str]:
    semantic_count = sum(1 for tok in tree_tokens if tok in ARITY)
    out = [HA_START]
    for i in range(semantic_count):
        out.append(PRIMITIVES[i % len(PRIMITIVES)])
    out.append(HA_END)
    return out


def tree_stats(tokens: list[str]) -> dict[str, int]:
    parsed = HACoTGrammar().parse(tokens)
    return {
        "nodes": parsed.node_count,
        "roots": len(parsed.roots),
        "max_depth": parsed.max_depth,
        "token_count": len(tokens),
        "size_bucket": min(15, int(math.log2(max(1, parsed.node_count))) + 1),
    }
