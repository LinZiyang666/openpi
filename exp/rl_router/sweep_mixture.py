#!/usr/bin/env python3
"""Measure success as a function of a FIXED teacher/cheap-arm mixture.

    python sweep_mixture.py --arm-yaml <train yaml> --split <split yaml> \
        --init-states-dir <B pool> --servers 127.0.0.1:8000 \
        --out-dir <dir> --episodes 200 --p 0.0 --p 0.15 --p 0.3 --p 0.5

Why this exists
---------------
The M6 runs all start at a 50/50 mixture and wander without learning to
discriminate: measured across 12,000 B-pool episodes with (task, init) fixed
effects and using only decisions that precede the outcome, the slope of success
in the teacher share is +0.004 +/- 0.011 -- indistinguishable from zero. Yet a
pure-student collection on the same pool scores 0.787 against the routed runs'
0.905. Both cannot be true of a straight line: success must rise steeply out of
p=0 and then flatten well before the region the router explores.

Where it flattens is the only number that matters for configuring this
experiment. It sets where a warm start should sit, how large lambda must be to
push the policy there, and -- if the knee turns out to be state-independent --
whether a *router* is the right object at all rather than a fixed ratio.

How
---
A router whose trunk is zeroed is a constant policy: ``relu(0 @ x + 0) = 0`` so
the logits are exactly ``b2``, whatever the observation. Setting ``b2`` to
``[0, logit]`` and sampling gives an exact, state-independent mixture. That is
the same degenerate shape the dead-trunk incident produced by accident; used
deliberately it is precisely the control this needs.

Each point runs ``episodes`` episodes with no dump (the features are not needed
and cost 4.5 MB each), then reports the realized arm rates and success.
"""
import argparse
import copy
import json
import math
import pathlib
import sys

import torch
import yaml as _yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openpi.cache.components.mlp_router_judge import ARM_SETS, save_router_weights  # noqa: E402


def constant_policy(out_path, *, p_teacher: float, arms: str, dim: int, hidden: int,
                    fields, dims, mu, sigma, cheap_arm: str = "student") -> str:
    """A router that ignores its input and samples ``p_teacher`` for teacher.

    The mixture is always teacher-vs-ONE cheap arm; ``cheap_arm`` names which.
    R_ts contrasts teacher with the distilled student, R_tc with the cache --
    and the cache is the arm the paper is actually about, so hard-coding
    ``student`` here made the instrument unable to measure the variant that
    matters. Any further arms are pushed to a logit of -30, i.e. excluded, so
    the realized mixture is exactly two-way whatever the arm set.
    """
    names = ARM_SETS[arms]
    for required in ("teacher", cheap_arm):
        if required not in names:
            raise SystemExit(
                f"arms={arms!r} has no {required!r} arm (it has {names}); "
                "the sweep measures teacher against exactly one cheap arm")
    if cheap_arm == "teacher":
        raise SystemExit("--cheap-arm cannot be 'teacher'; the sweep needs two distinct arms")
    b2 = torch.zeros(len(names))
    # softmax over two live rows: put the whole gap on the cheap arm's row.
    eps = 1e-6
    p = min(max(p_teacher, eps), 1 - eps)
    b2[names.index(cheap_arm)] = math.log((1 - p) / p)
    other = [i for i, n in enumerate(names) if n not in ("teacher", cheap_arm)]
    for i in other:
        b2[i] = -30.0                      # effectively excluded from the mixture
    save_router_weights(
        out_path, arms=arms, fields=fields, dims=dims, weights_version="v0",
        mu=mu, sigma=sigma,
        W1=torch.zeros(hidden, dim), b1=torch.zeros(hidden),
        W2=torch.zeros(len(names), hidden), b2=b2,
    )
    return str(out_path)


def sweep_yaml(arm_yaml, *, weights_path: str) -> dict:
    """The training yaml with its weights swapped and the dump switched off.

    Derived from the arm yaml rather than written fresh so the key builder,
    artifact and search config are provably the ones the M6 runs used -- the
    mixture has to be measured on the same observation space to be comparable.
    ``mode`` stays ``sample``: the point is a *mixture*, and argmax on a constant
    policy would collapse to a pure arm.
    """
    cfg = copy.deepcopy(_yaml.safe_load(pathlib.Path(arm_yaml).read_text(encoding="utf-8")))
    judge = cfg["checkpoints"]["cp1"]["judge"]
    judge["mode"] = "sample"
    judge["weights_path"] = weights_path
    judge.pop("dump_dir", None)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-yaml", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--init-states-dir", required=True)
    ap.add_argument("--servers", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference-weights", required=True,
                    help="a real warm-start; its fields/dims/mu/sigma are copied verbatim")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--gpu-ids", default="",
                    help="explicit CUDA device list, cycled across workers "
                         "(e.g. '7,7,7,4'); overrides --gpus — see worker_gpu_id")
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--episode-timeout-s", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, action="append", required=True,
                    help="teacher share to measure; repeat")
    ap.add_argument("--cheap-arm", default="student",
                    help="the arm teacher is mixed against: 'student' for R_ts, "
                         "'cache' for R_tc. Must exist in the checkpoint's arm set.")
    args = ap.parse_args()

    from openpi.cache.config import load_cache_config
    from openpi.conductor import ServerEndpoint, WorkerSpec

    from exp.rl_router.pilot_lambda import _endpoint, realized_teacher_rate
    from exp.rl_router.run_rl_router import (
        RouterBatchStrategy, btrain_pairs, make_slots, resolve_init_states_dir,
        run_round, sample_batch, worker_gpu_id,
    )

    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    init_dir = resolve_init_states_dir(args.init_states_dir)

    ref = torch.load(args.reference_weights, map_location="cpu", weights_only=False)
    meta = ref["meta"]
    arms, hidden = str(meta["arms"]), int(meta["hidden"])
    dim = int(ref["W1"].shape[1])

    pool = btrain_pairs(args.split)
    results = []
    for p in args.p:
        tag = f"p{p:.2f}".replace(".", "_")
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        wpath = constant_policy(
            d / "constant.pt", p_teacher=p, arms=arms, dim=dim, hidden=hidden,
            fields=tuple(meta["fields"]), dims=dict(meta["dims"]),
            mu=ref["feature_mu"], sigma=ref["feature_sigma"],
            cheap_arm=args.cheap_arm)
        ypath = d / "arm.yaml"
        ypath.write_text(_yaml.safe_dump(sweep_yaml(args.arm_yaml, weights_path=wpath),
                                         sort_keys=False), encoding="utf-8")
        load_cache_config(str(ypath))

        pairs = sample_batch(pool, batch_size=args.episodes, batch_idx=0, seed=args.seed)
        yaml_id = ypath.stem
        servers = [ServerEndpoint(*_endpoint(s)) for s in args.servers.split(",")]
        specs = [WorkerSpec(worker_id=f"w{i}", server_key=servers[i % len(servers)].key,
                            gpu_id=worker_gpu_id(i, gpus=args.gpus,
                                                 gpu_ids=args.gpu_ids),
                            conda_env=args.conda_env,
                            task_suite_name=args.suite, init_states_dir=init_dir)
                 for i in range(args.workers)]
        strategy = RouterBatchStrategy(
            suite=args.suite, yaml_path=str(ypath), run_id=f"sweep_{tag}",
            batch_id="sweep", weights_version="v0",
            bundle_id=f"sweep_{tag}", slots=make_slots(pairs, yaml_id=yaml_id),
            trials_per_task=args.episodes,
        )
        journal, rows = run_round(
            strategy=strategy, yaml_id=yaml_id, servers=servers, worker_specs=specs,
            journal_path=str(d / "journal.jsonl"), rows_path=str(d / "client_rows.jsonl"),
            bind_host=args.bind_host, episode_timeout_s=args.episode_timeout_s,
        )
        n = len(journal)
        ok = sum(1 for r in journal if r.get("success"))
        rate = realized_teacher_rate(rows)
        rec = {"p_target": p, "p_realized": rate, "episodes": n,
               "successes": ok, "success_rate": ok / n if n else None}
        (d / "result.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        results.append(rec)
        print(f"[sweep] p={p:.2f} realized={rate:.4f} success={ok}/{n}"
              f" = {rec['success_rate']:.4f}", flush=True)

    (out_root / "sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n p_target  p_realized  episodes  success")
    for r in results:
        print(f"  {r['p_target']:.2f}      {r['p_realized']:.4f}    {r['episodes']:5d}"
              f"    {r['success_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
