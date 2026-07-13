#!/usr/bin/env python
"""Torch-free boosting worker (avoids the PyTorch/XGBoost OpenMP segfault on macOS).

Run as a subprocess from run_baselines.py: it imports only numpy + xgboost/catboost
(never torch), so the two OpenMP runtimes never coexist in one process.

    python scripts/_boost_worker.py <npz_path> <kind> <out_json>
"""

import json
import sys

import numpy as np

# Importing this pulls in numpy/sklearn only — NOT torch (verified: no torch at
# import time anywhere on this path).
from defcon.eval.baselines import train_boosting


def main() -> None:
    npz_path, kind, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    d = np.load(npz_path)
    m = train_boosting(d["Xtr"], d["ytr"], d["Xva"], d["yva"], kind=kind, seed=int(d["seed"]))
    with open(out_json, "w") as f:
        json.dump(m, f)


if __name__ == "__main__":
    main()
