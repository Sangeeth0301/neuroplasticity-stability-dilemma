"""Build the cached scorer datasets that every training and evaluation script loads.

    python scripts/pipeline/build_scoring_cache.py [--train-threads 12] [--test-threads 6]

For every probe, the memory returns the top-M earlier candidates and each (probe, candidate)
pair is turned into a 1024-number unit-length feature vector (see src/qcs/encode/assemble.py).
Two files are written to data/processed/ (both are git-ignored because they are large and
fully regenerable):

    scoring_train.npz   12 training threads -> 720 probes x 64 candidates   (the 120-task stream)
    scoring_test.npz     6 sealed test threads -> 360 probes x 64 candidates (final numbers only)

The test file is built here but is only ever read by the final evaluation scripts; no setting
is tuned on it.
"""
from __future__ import annotations

import argparse

from qcs.config import Config
from qcs.train.dataset import ScoringDataset

ap = argparse.ArgumentParser()
ap.add_argument("--config", nargs="*", default=["tuned"])
ap.add_argument("--train-threads", type=int, default=12)
ap.add_argument("--test-threads", type=int, default=6)
ap.add_argument("--candidates", type=int, default=64)
args = ap.parse_args()

cfg = Config.load(*args.config)
cfg.validate()

for split, n in (("train", args.train_threads), ("test", args.test_threads)):
    ds = ScoringDataset(cfg, split=split).build(n_threads=n, max_candidates=args.candidates, verbose=False)
    path = ds.save()
    print(f"{split}: {len(ds.examples)} probes -> {path}")
    print("   ", ds.stats())
