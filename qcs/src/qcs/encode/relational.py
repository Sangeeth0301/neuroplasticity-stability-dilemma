"""The ρ block: everything about a (query, candidate, user) triple that is not semantics.

THE RULE THAT MATTERS: every feature here must already be bounded to [-1, 1] before it
reaches assembly. Amplitude encoding normalises the full 1024-vector onto the unit
sphere, which destroys magnitude information — so a raw count or a raw age in seconds
does not survive the trip into Hilbert space. Magnitudes are therefore encoded as
bounded ratios or as angles (sin/cos of a log-scaled quantity), never as raw numbers.

Layout (256 dims):
    [  0: 64]  user salience        projected EMA of the asking participant's turns
    [ 64: 80]  recency / temporal   log-distance, sin/cos phase, same-session flags
    [ 80:112]  graph features       supersession, contradiction, hops (Phase 3 fills these)
    [112:128]  speaker features     same-speaker, is-owner, one-hot of speaker index
    [128:136]  usage features       times previously selected, bounded
    [136:256]  reserve              deliberate zero-padding for later features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SALIENCE = slice(0, 64)
RECENCY = slice(64, 80)
GRAPH = slice(80, 112)
SPEAKER = slice(112, 128)
USAGE = slice(128, 136)
RESERVE = slice(136, 256)
RHO_DIM = 256

MAX_SPEAKERS = 8


@dataclass
class GraphFeatures:
    """Filled by the memory manifold in Phase 3; zeros are a valid 'unknown'."""

    is_superseded: float = 0.0
    supersedes_something: float = 0.0
    contradicted: float = 0.0
    shares_entity_with_query: float = 0.0
    hops_to_query: float = 0.0          # bounded: 1/(1+hops)
    degree: float = 0.0                 # bounded: tanh(deg/8)

    def vector(self) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        v[:6] = [self.is_superseded, self.supersedes_something, self.contradicted,
                 self.shares_entity_with_query, self.hops_to_query, self.degree]
        return v


def recency_features(turn_gap: int, seconds_gap: float) -> np.ndarray:
    """Bounded encodings of 'how long ago'.

    A raw gap of 500 turns would dominate the vector and then be destroyed by
    normalisation. log-compression plus sin/cos phases keeps the information and the
    scale.
    """
    v = np.zeros(16, dtype=np.float32)
    g = max(int(turn_gap), 0)
    lg = np.log1p(g)
    v[0] = np.tanh(lg / 6.0)                      # smooth, saturating distance
    v[1] = 1.0 / (1.0 + g)                        # sharp near-field signal
    v[2] = float(g <= 10)
    v[3] = float(10 < g <= 50)
    v[4] = float(50 < g <= 200)
    v[5] = float(g > 200)
    for i, period in enumerate((8.0, 32.0, 128.0, 512.0)):
        v[6 + 2 * i] = np.sin(g / period)
        v[7 + 2 * i] = np.cos(g / period)
    v[14] = np.tanh(np.log1p(max(seconds_gap, 0.0)) / 12.0)
    v[15] = 1.0
    return v


def speaker_features(
    query_user: str, cand_user: str, owner_user: str | None, users: list[str]
) -> np.ndarray:
    v = np.zeros(16, dtype=np.float32)
    v[0] = float(query_user == cand_user)
    v[1] = float(owner_user is not None and query_user == owner_user)
    v[2] = float(owner_user is not None and cand_user == owner_user)
    idx = users.index(cand_user) if cand_user in users else -1
    if 0 <= idx < MAX_SPEAKERS:
        v[3 + idx] = 1.0
    return v


def usage_features(times_selected: int, times_seen: int) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[0] = np.tanh(times_selected / 4.0)
    v[1] = np.tanh(times_seen / 16.0)
    v[2] = float(times_selected > 0)
    v[3] = times_selected / max(times_seen, 1)
    return v


def build_rho(
    salience: np.ndarray,
    turn_gap: int,
    seconds_gap: float,
    query_user: str,
    cand_user: str,
    owner_user: str | None,
    users: list[str],
    graph: GraphFeatures | None = None,
    times_selected: int = 0,
    times_seen: int = 0,
) -> np.ndarray:
    rho = np.zeros(RHO_DIM, dtype=np.float32)
    s = np.asarray(salience, dtype=np.float32)[:64]
    rho[SALIENCE][: len(s)] = np.tanh(s)
    rho[RECENCY] = recency_features(turn_gap, seconds_gap)
    rho[GRAPH] = (graph or GraphFeatures()).vector()
    rho[SPEAKER] = speaker_features(query_user, cand_user, owner_user, users)
    rho[USAGE] = usage_features(times_selected, times_seen)
    assert np.all(np.abs(rho) <= 1.0 + 1e-6), "ρ feature outside [-1,1] — see module docstring"
    return rho
