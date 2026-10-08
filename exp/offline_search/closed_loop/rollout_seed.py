"""Explicit opt-in policy process RNG initialization for rollout replicates.

This changes random draws, not the keyed R11 coin or the init selection. Streams
remain dependent on warmup and concurrent policy request order. Equal seeds on
different server processes can start equal streams; no independence is claimed.
"""
def seed_policy_process(seed):
    import random
    import numpy as np
    import torch

    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError('rollout seed must be in [0,2**32)')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return dict(seed=seed, python=True, numpy=True, torch=True,
                scope='process; warmup/concurrent-request-order dependent', knob_coin_seed=0)
