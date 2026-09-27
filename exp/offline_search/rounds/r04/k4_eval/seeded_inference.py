"""Plain B0 proposal with explicit process RNG initialization for all-MISS controls.

Use only with periodic:1 / step0=judge. chain.sh supplies --os-seed = base*65536 + port.
Seeding occurs once before the first proposal/MISS, including with a reused pickle.
Concurrent request ordering still makes a stochastic rollout scheduling-dependent.
"""
import argparse
import os
import random
import sys
import threading

import numpy as np
from exp.offline_search.harness.baselines import B0Current

_seed_lock = threading.Lock()
_seeded = None


def seed_process():
    global _seeded
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--os-seed", type=int, default=0)
    opts, _ = parser.parse_known_args(sys.argv[1:])
    identity = (os.getpid(), opts.os_seed)
    with _seed_lock:
        if _seeded != identity:
            import torch
            random.seed(opts.os_seed)
            np.random.seed(opts.os_seed % (2 ** 32))
            torch.manual_seed(opts.os_seed)
            _seeded = identity
            print(f"OSCL_POLICY_SEED pid={identity[0]} seed={identity[1]}", flush=True)
    return opts.os_seed


class SeededInference(B0Current):
    def query(self, q):
        seed_process()
        return super().query(q)
