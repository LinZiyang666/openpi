"""One-off precompute (optional cache) of the current-library PCA-64 bases used by AWM with fit_data on the current
library. Same function (awm.pca_fit) and single-threaded BLAS as the in-fit computation, so the cached arrays equal
what fit() would compute; fit() loads them when meta.json matches the library (n, sha1 of ids.json), else it computes
in place. Output: <DERIVED>/pca_current/<m>_<s>/<field>/{mean,basis,proj}.npy + meta.json.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <cpus> .venv/bin/python \
        exp/offline_search/rounds/r02/g1_awm/pca_current.py [pi05_spatial,...] [field]
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, "/home/weiland/projects/openpi")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from exp.offline_search.harness import store  # noqa: E402
from exp.offline_search.rounds.r02.g1_awm.awm import PCA_CUR, lib_fingerprint, pca_fit  # noqa: E402

ROOT = "/dev/shm/offline_search_store"


def main():
    keys = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
    fields = [sys.argv[2]] if len(sys.argv) > 2 else ["v0", "v1"]
    for key in keys:
        L = store.LibraryView(ROOT, key, "current")
        for f in fields:
            t0 = time.time()
            mu, B, P = pca_fit(getattr(L, f"key_{f}"))
            d = PCA_CUR / key / f
            d.mkdir(parents=True, exist_ok=True)
            for name, a in (("mean", mu), ("basis", B), ("proj", P)):
                np.save(d / f"{name}.tmp.npy", a)
                os.replace(d / f"{name}.tmp.npy", d / f"{name}.npy")
            (d / "meta.json").write_text(json.dumps({"key": key, "field": f, **lib_fingerprint(L),
                                                     "fit_s": time.time() - t0}, indent=1))
            print(key, f, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
