"""Ansatz topology: which qubit pairs each layer touches.

Brickwall alternates even and odd nearest-neighbour pairs; ladder walks down the
register. Both are built from the same general SU(4) block, so depth and parameter count
are directly comparable between them.

The light-cone consequence, measured before this repo existed and now enforced in
`Config.validate()`: a parameter acting on qubits (0,1) needs roughly N/2 brickwall
layers before it can influence a single-qubit readout at the far end. Below that depth a
large fraction of parameters are structurally untrainable — their gradient is not small,
it is exactly zero.
"""

from __future__ import annotations


def brickwall_pairs(n: int) -> list[tuple[int, int]]:
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    return even + odd


def ladder_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def ring_pairs(n: int) -> list[tuple[int, int]]:
    return brickwall_pairs(n) + [(n - 1, 0)]


LAYOUTS = {"brickwall": brickwall_pairs, "ladder": ladder_pairs, "ring": ring_pairs}


def pairs_for(name: str, n: int) -> list[tuple[int, int]]:
    if name not in LAYOUTS:
        raise KeyError(f"unknown ansatz '{name}'; have {sorted(LAYOUTS)}")
    return LAYOUTS[name](n)


def n_params(name: str, n: int, depth: int) -> int:
    """|θ| = depth × gates_per_layer × 15."""
    return depth * len(pairs_for(name, n)) * 15


def light_cone_depth(n: int, readout_qubit: int, name: str = "brickwall") -> int:
    """Layers needed before a gate on qubits (0,1) can reach `readout_qubit`.

    Computed by propagating the influenced-qubit set through the layer pattern, so it
    stays correct if the topology changes.
    """
    pairs = pairs_for(name, n)
    influenced = {0, 1}
    for depth in range(1, 4 * n + 1):
        for a, b in pairs:
            if a in influenced or b in influenced:
                influenced |= {a, b}
        if readout_qubit in influenced:
            return depth
    return -1
