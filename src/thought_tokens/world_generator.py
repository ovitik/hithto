from __future__ import annotations

from dataclasses import dataclass, asdict
import random
from typing import Literal


TaskType = Literal["nesting", "temporal", "causal", "plan", "belief"]
SplitName = Literal["train", "validation", "paraphrase", "new_combinations", "ood_depth", "adversarial"]


@dataclass
class WorldExample:
    id: str
    split: str
    task_type: str
    depth: int
    text: str
    question: str
    answer: str
    symbolic: dict
    paraphrase_id: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


class CzechWorldGenerator:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.objects = ["klíč", "dopis", "náramek", "lístek", "sešit", "mobil"]
        self.containers = ["krabička", "zásuvka", "batoh", "skříňka", "taška", "kufr"]
        self.places = ["kuchyně", "pracovna", "chodba", "ložnice", "garáž", "kancelář"]
        self.names = ["Eva", "Petr", "Jana", "Marek", "Lucie", "Tomáš"]

    def generate(self, split: SplitName = "train", n: int = 100) -> list[WorldExample]:
        examples: list[WorldExample] = []
        for i in range(n):
            task_type = self._task_type_for_index(i)
            if task_type == "nesting":
                ex = self._nesting(split, i)
            elif task_type == "temporal":
                ex = self._temporal(split, i)
            elif task_type == "causal":
                ex = self._causal(split, i)
            elif task_type == "plan":
                ex = self._plan(split, i)
            else:
                ex = self._belief(split, i)
            examples.append(ex)
        return examples

    def _task_type_for_index(self, i: int) -> TaskType:
        # First two types are fully varied; later types are intentionally simpler scaffolds.
        return ["nesting", "temporal", "nesting", "temporal", "causal", "plan", "belief"][i % 7]  # type: ignore[return-value]

    def _depth(self, split: str) -> int:
        if split == "ood_depth":
            return self.rng.randint(4, 8)
        if split == "adversarial":
            return self.rng.randint(2, 5)
        return self.rng.randint(1, 3)

    def _nesting(self, split: str, i: int) -> WorldExample:
        depth = self._depth(split)
        obj = self.rng.choice(self.objects)
        chain = self.rng.sample(self.containers + self.places, k=depth + 1)
        statements = []
        current = obj
        for nxt in chain:
            statements.append(self._location_sentence(current, nxt))
            current = nxt
        if split == "adversarial":
            statements.insert(0, self._location_sentence(self.rng.choice(self.objects), self.rng.choice(self.places)))
        self.rng.shuffle(statements) if split == "paraphrase" else None
        question = f"Kde je nakonec {obj}?"
        answer = chain[-1]
        text = " ".join(statements)
        return WorldExample(
            id=f"{split}-nesting-{i}",
            split=split,
            task_type="nesting",
            depth=depth,
            text=text,
            question=question,
            answer=answer,
            symbolic={"object": obj, "chain": chain},
            paraphrase_id=f"nesting-{obj}-{answer}" if split == "paraphrase" else None,
        )

    def _location_sentence(self, item: str, place: str) -> str:
        templates = [
            "{item} je v {place}.",
            "Do {place} byl uložen {item}.",
            "Uvnitř {place} se nachází {item}.",
            "{item} skončil v {place}.",
        ]
        return self.rng.choice(templates).format(item=item, place=place)

    def _temporal(self, split: str, i: int) -> WorldExample:
        depth = self._depth(split)
        actor = self.rng.choice(self.names)
        events = [
            "zapnul počítač",
            "odeslal zprávu",
            "zavolal na nádraží",
            "vyzvedl balík",
            "zamkl dveře",
            "odešel do práce",
            "zkontroloval kalendář",
            "připravil snídani",
        ]
        chosen = self.rng.sample(events, k=depth + 1)
        statements = []
        for prev, nxt in zip(chosen, chosen[1:]):
            if self.rng.random() < 0.5:
                statements.append(f"{actor} nejdřív {prev} a potom {nxt}.")
            else:
                statements.append(f"Než {actor} {nxt}, {prev}.")
        if split == "adversarial":
            statements.append(f"{self.rng.choice(self.names)} mezitím {self.rng.choice(events)}.")
        question = f"Co {actor} udělal jako poslední?"
        answer = chosen[-1]
        return WorldExample(
            id=f"{split}-temporal-{i}",
            split=split,
            task_type="temporal",
            depth=depth,
            text=" ".join(statements),
            question=question,
            answer=answer,
            symbolic={"actor": actor, "events": chosen},
            paraphrase_id=f"temporal-{actor}-{answer}" if split == "paraphrase" else None,
        )

    def _causal(self, split: str, i: int) -> WorldExample:
        depth = self._depth(split)
        text = (
            "Vlak má zpoždění. Přestup má jen malou rezervu. "
            "Když je zpoždění větší než rezerva, cestující spoj nestihne."
        )
        return WorldExample(
            id=f"{split}-causal-{i}",
            split=split,
            task_type="causal",
            depth=depth,
            text=text,
            question="Stihne cestující navazující spoj?",
            answer="ne",
            symbolic={"delay": True, "small_buffer": True},
        )

    def _plan(self, split: str, i: int) -> WorldExample:
        depth = self._depth(split)
        name = self.rng.choice(self.names)
        text = (
            f"{name} chtěla jet autobusem do centra a potom dojít do muzea. "
            "Autobus byl zrušen, takže musí nejdřív zvolit jinou dopravu."
        )
        return WorldExample(
            id=f"{split}-plan-{i}",
            split=split,
            task_type="plan",
            depth=depth,
            text=text,
            question="Který krok plánu je nutné upravit?",
            answer="dopravu do centra",
            symbolic={"changed_step": "transport"},
        )

    def _belief(self, split: str, i: int) -> WorldExample:
        depth = self._depth(split)
        text = (
            "Klíč je ve skříňce. Eva viděla, že Petr přesunul klíč do batohu. "
            "Petr si myslí, že Eva přesun neviděla."
        )
        return WorldExample(
            id=f"{split}-belief-{i}",
            split=split,
            task_type="belief",
            depth=depth,
            text=text,
            question="Kde si Petr myslí, že Eva hledá klíč?",
            answer="ve skříňce",
            symbolic={"real": "batoh", "petr_believes_eva_believes": "skříňka"},
        )
