"""Label space and persistent node identifiers (Section 3.1, Eq. 5)."""

from __future__ import annotations

from typing import Sequence


def ip_to_int(ip_str: str) -> int:
    """Map an IPv4 address a.b.c.d to a·256³ + b·256² + c·256 + d  (Eq. 5).

    The identifier is used only as a memory key and is never fed to the
    network as a feature. Non-IPv4 strings fall back to a masked hash.
    """
    try:
        val = 0
        for p in str(ip_str).split("."):
            val = val * 256 + int(p)
        return val & 0x7FFFFFFFFFFFFFFF
    except Exception:
        return abs(hash(str(ip_str))) & 0x7FFFFFFFFFFFFFFF


class LabelSpace:
    """Ordered class list with stable name <-> id mapping.

    Index 0 must be the benign class.
    """

    def __init__(self, classes: Sequence[str], benign_label: str = "benign"):
        self.classes = [c.lower() for c in classes]
        self.benign_label = benign_label.lower()
        if self.classes[0] != self.benign_label:
            raise ValueError("The first class must be the benign class.")
        self.attack2id = {a: i for i, a in enumerate(self.classes)}
        self.id2attack = {i: a for a, i in self.attack2id.items()}

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def attack_ids(self) -> list[int]:
        return [i for i, a in self.id2attack.items() if a != self.benign_label]

    def to_id(self, label: str) -> int:
        """Unknown labels map to benign (as in the original pipeline)."""
        return self.attack2id.get(str(label).lower(), self.attack2id[self.benign_label])
