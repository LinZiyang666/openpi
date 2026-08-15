"""Warm-start collection pass: one constant arm over B-train (§3.8, M5b).

This is the pass that breaks the circular dependency. The router's encoder v1
needs robot_state statistics; the statistics come from collected features; and
collecting features normally needs a router. A ``constant_arm`` judge cuts the
loop: it makes no decision, needs no weights and no statistics, and still emits
the v0-raw features and the episode outcomes the success head is fit on.

The arm is the student (``R_ts`` config): that is the arm whose success the
head predicts, and hence the one the graft initialises. R_tc / R_tsc's cache
arm has no equivalent head — the pre-registered disclosure says so rather than
inventing an initialisation for it.

Collection runs at the full B-train size (450 episodes/suite) as ONE logical
batch, so every episode lands under one ``<run_id>/<batch_id>`` shard directory.

Usage (conductor host)::

    PYTHONPATH=. uv run exp/rl_router/collect_warmstart.py \\
        --suite libero_10 --split exp/ablation_study/config/common/split_libero_10.yaml \\
        --arm-yaml exp/rl_router/config/libero_10/collect_student.yaml \\
        --run-id ws_l10 --servers <host>:8000 --workers 8 \\
        --out-dir exp/rl_router/data/warmstart/l10
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pathlib

import yaml

from openpi.cache.config import load_cache_config
from openpi.conductor import ServerEndpoint, WorkerSpec

from exp.rl_router.run_rl_router import (
    RouterBatchStrategy,
    btrain_pairs,
    make_slots,
    resolve_init_states_dir,
    run_round,
)

logger = logging.getLogger(__name__)

# The collection batch id. One directory for the whole pass keeps the fitting
# step a single manifest read.
COLLECT_BATCH_ID = "collect"


def build_collect_yaml(arm_yaml: str | pathlib.Path, *, arm: str, dump_dir: str) -> dict:
    """Turn a router arm yaml into its constant-arm collection twin.

    Derived from the arm yaml rather than written independently so the key
    builder, the artifact and the search config are provably the same ones the
    RL run will use — the features must come from the same observation space.
    """
    cfg = copy.deepcopy(yaml.safe_load(
        pathlib.Path(arm_yaml).read_text(encoding="utf-8")
    ))
    judge = cfg["checkpoints"]["cp1"]["judge"]
    judge.pop("weights_path", None)
    judge.pop("temperature", None)
    judge.pop("seed", None)
    judge["constant_arm"] = arm
    judge["mode"] = "argmax"
    judge["dump_dir"] = dump_dir
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--arm-yaml", required=True, help="the R_ts arm yaml to derive from")
    parser.add_argument("--arm", default="student", choices=["teacher", "student", "cache"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--servers", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--init-states-dir", required=True,
                        help="B-train diff pool; empty would silently collect on the A pool")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--conda-env", default="")
    parser.add_argument("--episode-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = pathlib.Path(args.shard_root) / args.run_id / COLLECT_BATCH_ID

    collect_yaml = out_dir / "collect.yaml"
    cfg = build_collect_yaml(args.arm_yaml, arm=args.arm, dump_dir=args.shard_root)
    collect_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    load_cache_config(str(collect_yaml))  # allowlist + mlp_router rules, before any episode
    logger.info("collection yaml: %s (constant_arm=%s)", collect_yaml, args.arm)

    # Same guard as the training loop: an empty init dir makes LIBERO fall back
    # to the official pruned_init A pool, so the warm-start head would be fit on
    # the frozen test set.
    init_dir = resolve_init_states_dir(args.init_states_dir)

    pairs = btrain_pairs(args.split)
    yaml_id = collect_yaml.stem
    slots = make_slots(pairs, yaml_id=yaml_id)
    servers = [ServerEndpoint(*_endpoint(s)) for s in args.servers.split(",")]
    specs = [
        WorkerSpec(worker_id=f"w{i}", server_key=servers[i % len(servers)].key,
                   gpu_id=str(i % args.gpus), conda_env=args.conda_env,
                   task_suite_name=args.suite, init_states_dir=init_dir)
        for i in range(args.workers)
    ]
    strategy = RouterBatchStrategy(
        suite=args.suite, yaml_path=str(collect_yaml), run_id=args.run_id,
        batch_id=COLLECT_BATCH_ID, weights_version=f"constant-{args.arm}",
        bundle_id=f"ws_{args.run_id}", slots=slots, trials_per_task=len(pairs),
    )
    journal_path = str(out_dir / "journal.jsonl")
    journal_rows, client_rows = run_round(
        strategy=strategy, yaml_id=yaml_id, servers=servers, worker_specs=specs,
        journal_path=journal_path, rows_path=str(out_dir / "per_step_rows.jsonl"),
        bind_host=args.bind_host, episode_timeout_s=args.episode_timeout_s,
    )
    # The expected set is what fit_warmstart admits against: an episode that
    # never produced a journal row at all has to count against the failure rate,
    # and it can only be seen by comparing to the dispatched slots.
    (out_dir / "expected_slots.json").write_text(
        json.dumps([uid for _, _, uid in slots], indent=2), encoding="utf-8"
    )
    logger.info(
        "collection done: %d journal rows, %d client rows, shards under %s. "
        "Next: fit_warmstart.py",
        len(journal_rows), len(client_rows), shard_dir,
    )


def _endpoint(spec: str) -> tuple[str, int]:
    host, port = spec.rsplit(":", 1)
    return host, int(port)


if __name__ == "__main__":
    main()
