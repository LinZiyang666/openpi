"""ROUND 4 end-to-end build equivalence: orig vs avgpool keybuilder through build()+search.

For a subset of real fake-stage1 steps, runs the FULL src build() path on both the original
CP1SpatialPool16KeyBuilder and the R4 CP1SpatialPool16AvgPoolKeyBuilder, then:
  (1) compares every query_keys field bit/numerically (vision_0, vision_1, robot_state);
  (2) feeds each query into the real backend.search(spec) (depth_1.yaml params) and checks
      the 3-way verdict (FULL/WARM/MISS) + winner id do not flip.

Hard verdict (blocking): query-key max|diff| == 0 (bit-identical) AND verdict_flips == 0
AND winner_mismatch == 0.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r4_equiv_endtoend.py \
        --cache-config exp/cache_latency_bench/config/depth_study/depth_1.yaml \
        --max-steps 200
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os

import h5py
import torch
import tyro

from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.config import build_cache_components, load_cache_config
from openpi.cache.storage_types import QueryFilter, QuerySpec
from openpi.cache.types import CheckpointID

from exp.cache_latency_bench.opt.r4_build_keybuilder import CP1SpatialPool16AvgPoolKeyBuilder
from exp.common.build_in_memory_cache_artifact import _build_fake_stage1

# depth_1.yaml 3-way thresholds + search params.
FULL = 0.997697
WARM = 0.997403
W = {"vision_0": 0.25, "vision_1": 0.4375, "robot_state": 0.3125}
FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "vision_1": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}
SCORE_NORM = {
    "type": "per_field",
    "fields": {
        "vision_0": {"method": "zscore", "params": {"mu": 0.9739899923664463, "sigma": 0.0061831533438692935, "squash": "tanh"}},
        "vision_1": {"method": "zscore", "params": {"mu": 0.9659078322399228, "sigma": 0.006527797454113087, "squash": "tanh"}},
        "robot_state": {"method": "zscore", "params": {"mu": -1.9584325681212513, "sigma": 0.7484941685797242, "squash": "tanh"}},
    },
}


def verdict3(score) -> str:
    if score is None:
        return "MISS"
    if score >= FULL:
        return "FULL"
    if score >= WARM:
        return "WARM"
    return "MISS"


@dataclasses.dataclass
class Args:
    cache_config: str = "exp/cache_latency_bench/config/depth_study/depth_1.yaml"
    h5_dir: str = "exp/common/data/db/libero_cache/libero_10"
    out: str = "exp/cache_latency_bench/data/opt_bench/r4_equiv_result.json"
    max_steps: int = 200


def iter_steps(h5_dir: str, max_steps: int):
    paths = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    n = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            task = str(f.attrs.get("task", ""))
            names = sorted((k for k in f.keys() if k.startswith("step_")),
                           key=lambda s: int(s.split("_")[-1]))
            for name in names:
                yield task, _build_fake_stage1(f[name])
                n += 1
                if n >= max_steps:
                    return


def build_keys(kb, fake_stage1):
    kb.collect(CheckpointID.CP1, stage1=fake_stage1)
    keys = kb.build(CheckpointID.CP1)
    kb.clear()
    return keys


def main(args: Args) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.set_grad_enabled(False)

    config = load_cache_config(args.cache_config)
    components = build_cache_components(config)
    orig_kb = components["key_builder"]
    assert isinstance(orig_kb, CP1SpatialPool16KeyBuilder)
    backend = components["storage"]._backend

    opt_kb = CP1SpatialPool16AvgPoolKeyBuilder(enabled_fields=None)
    opt_kb._enabled = orig_kb._enabled  # mirror the active-field set

    keydiff_max = 0.0
    keybit_ne = 0
    nq = 0
    verdict_flips = 0
    winner_mismatch = 0
    examples = []

    for task, fs in iter_steps(args.h5_dir, args.max_steps):
        ko = build_keys(orig_kb, fs)
        kn = build_keys(opt_kb, fs)

        # (1) field-wise key equivalence
        assert set(ko) == set(kn), (set(ko), set(kn))
        for fld in ko:
            d = (ko[fld] - kn[fld]).abs().max().item()
            keydiff_max = max(keydiff_max, d)
            keybit_ne += int((ko[fld] != kn[fld]).sum().item())

        # (2) downstream search verdict
        def run(query_keys):
            spec = QuerySpec(
                query_keys={f: query_keys[f] for f in W if f in query_keys},
                top_k=1,
                checkpoint_id=CheckpointID.CP1,
                filters=QueryFilter(task_key=task),
                fusion_weights=W,
                fusion_method="weighted_score_sum",
                field_similarity=FIELD_SIM,
                score_normalization=SCORE_NORM,
            )
            r = backend.search(spec)
            return (r[0].id, r[0].score) if r else (None, None)

        io, so = run(ko)
        i_n, sn = run(kn)
        nq += 1
        if io != i_n:
            winner_mismatch += 1
        if verdict3(so) != verdict3(sn):
            verdict_flips += 1
            if len(examples) < 8:
                examples.append({"task": task[:30], "so": so, "sn": sn})

    result = {
        "n_steps": nq,
        "query_key_max_abs_diff": keydiff_max,
        "query_key_bit_differing_total": keybit_ne,
        "query_key_bit_identical": keybit_ne == 0,
        "verdict_flips_3way": verdict_flips,
        "winner_id_mismatch": winner_mismatch,
        "PASS": (keybit_ne == 0 and verdict_flips == 0 and winner_mismatch == 0),
        "flip_examples": examples,
        "cache_config": args.cache_config,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"[r4-equiv] {'PASS' if result['PASS'] else 'FAIL'} -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
