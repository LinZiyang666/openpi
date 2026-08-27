"""Isolated cost benchmark (compute / latency axes) for the precheck arms.

Two passes over the SAME (block, init-set, arm-order) manifest, each against
its own immutable server launch (``OPENPI_MONITOR_LEVEL`` is process-cached):

  pass A (``--bench-pass compute``): server launched with SNAPSHOT. Per
    (arm, block) unit the runner clears the server metrics, runs the unit's
    10 episodes serially (single conductor worker — the server sees only this
    unit's traffic, so the clear/dump window attributes rows exactly), then
    dumps-and-clears and keeps the CUDA stage timing rows. The response's
    ``level`` must be SNAPSHOT.
  pass B (``--bench-pass latency``): server launched with OFF. No metrics API
    is touched (OFF makes ``dump_metrics`` an error by design); the latency
    sample is the client-side per-episode ``infer_ms / infers`` accumulator,
    aggregated to a block mean. Server metadata must attest
    ``monitor_level == OFF``.

Sampling (plan 4.6): block b uses A' init index b-1 of every task
(task-stratified rotation, zero freedom); block R+... never touches the last
A' init, which is reserved for the per-(arm, pass) warmup episode (discarded;
run before the first measured block so bundle load + compile transients stay
out of every window). Arm order within a block is a seed-fixed shuffle shared
by both passes.

Unit execution is delegated to ``run_precheck`` (single arm, --trials 1,
single worker) so the verdict path, pairing discipline and launch-contract
assertions are byte-identical to the SR run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import subprocess
import sys

import torch
import websockets.sync.client

import msgpack_numpy

CORE_ARMS = ("dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10", "dsp_s0", "dsp_sv")
WARMUP_INIT_POS = -1  # last A' init per task, never used by measured blocks


def _ctrl(server: str, payload: dict) -> dict:
    """Send one ctrl message on a fresh connection and return the response."""
    host, port = server.rsplit(":", 1)
    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(
        f"ws://{host}:{port}", max_size=None, open_timeout=30,
    ) as ws:
        ws.recv()  # metadata frame
        ws.send(packer.packb(payload))
        return msgpack_numpy.unpackb(ws.recv())


def assert_compute_pass_metadata(meta: dict) -> None:
    """Refuse a compute pass unless every stage probe is a real CUDA event.

    ``cuda_available`` alone cannot distinguish an explicit CPU/meta stage on
    a CUDA-capable host; the server-attested per-stage probe-backend map is
    the authority (G2R3-B3).
    """
    if meta.get("cuda_available") is not True:
        raise SystemExit(
            "compute pass requires server-attested cuda_available=True; "
            f"metadata says {meta.get('cuda_available')!r}"
        )
    required = {"stage1", "stage2", "stage3"}
    backends = meta.get("stage_probe_backends")
    if not isinstance(backends, dict) or set(backends) != required:
        raise SystemExit(
            "compute pass requires an exact stage1/stage2/stage3 probe-backend "
            f"attestation; got {backends!r}"
        )
    bad = {k: v for k, v in backends.items() if v != "cuda"}
    if bad:
        raise SystemExit(
            f"compute pass requires CUDA-event probes on all stages; server "
            f"attests non-CUDA backends: {bad} — CPU wall time must never be "
            "adjudicated as GPU compute"
        )
    devices = meta.get("stage_devices")
    if not isinstance(devices, dict) or set(devices) != required:
        raise SystemExit(
            "compute pass requires exact effective stage_devices for all three stages"
        )
    bad_devices = {k: v for k, v in devices.items()
                   if not isinstance(v, str) or not v.startswith("cuda")}
    if bad_devices:
        raise SystemExit(
            f"compute pass effective stages are not all CUDA devices: {bad_devices}"
        )


def _server_metadata(server: str) -> dict:
    host, port = server.rsplit(":", 1)
    with websockets.sync.client.connect(
        f"ws://{host}:{port}", max_size=None, open_timeout=30,
    ) as ws:
        return msgpack_numpy.unpackb(ws.recv())


def materialize_block_pools(
    aprime_dir: pathlib.Path, out_root: pathlib.Path, blocks: int, seed: int,
) -> tuple[list[pathlib.Path], dict, str]:
    """Block pools by pre-registered fixed-seed per-task permutation (G2-B5).

    For every task the measurable A' subset positions (all but the
    warmup-reserved last one) are permuted once with a task-derived fixed
    seed; block b takes permutation[b]. Returns (pools, mapping, digest) —
    the mapping records the subset position each (task, block) used, and the
    digest binds the two passes to identical pools.
    """
    import random as _random

    pools = []
    init_files = sorted(aprime_dir.glob("*.init"))
    if not init_files:
        raise SystemExit(f"no .init files under {aprime_dir}")
    mapping: dict[str, dict[str, int]] = {}
    for t_idx, f in enumerate(init_files):
        states = torch.load(f, weights_only=False)
        usable = len(states) - 1  # last position is warmup-reserved
        if blocks > usable:
            raise SystemExit(
                f"{f}: {blocks} blocks need {blocks} distinct inits but only "
                f"{usable} are measurable (last one is warmup-reserved)"
            )
        perm = list(range(usable))
        _random.Random(seed * 10_000 + t_idx).shuffle(perm)
        mapping[f.name] = {str(b): perm[b] for b in range(blocks)}
    for b in range(blocks):
        pool_dir = out_root / f"block_{b:02d}"
        pool_dir.mkdir(parents=True, exist_ok=True)
        for f in init_files:
            states = torch.load(f, weights_only=False)
            pos = mapping[f.name][str(b)]
            torch.save(states[pos:pos + 1], pool_dir / f.name)
        pools.append(pool_dir)
    warmup_dir = out_root / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    for f in init_files:
        states = torch.load(f, weights_only=False)
        torch.save(states[WARMUP_INIT_POS:], warmup_dir / f.name)
    # The digest covers the ACTUAL state bytes of every materialised block
    # pool file plus the official-position mapping — two different A' trees
    # with equal filenames/lengths no longer collide (G2R2-B5).
    h = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode("utf-8"))
    for pool_dir in pools:
        for f in sorted(pool_dir.glob("*.init")):
            h.update(f.name.encode("utf-8"))
            h.update(f.read_bytes())
    return pools, mapping, h.hexdigest()


def run_unit(args, arm: str, pool_dir: pathlib.Path, tag: str) -> pathlib.Path:
    """One (arm, block|warmup) unit through run_precheck: 10 tasks x 1 init."""
    unit_dir = pathlib.Path(args.out_dir) / "units" / tag
    unit_dir.mkdir(parents=True, exist_ok=True)
    # The A' record's digests describe the full pool; block pools are derived
    # slices, so unit runs pass a slice-local record generated on the fly.
    record = {"suite": args.task_suite, "apool_dir": str(pool_dir), "files": {}}
    record_path = unit_dir / "pool_record.yaml"
    import hashlib

    import yaml as _yaml

    rollup = hashlib.sha256()
    for f in sorted(pool_dir.glob("*.init")):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        record["files"][f.name] = {"sha256": digest, "count": 1}
        rollup.update(digest.encode())
    record["rollup_sha256"] = rollup.hexdigest()
    record_path.write_text(_yaml.safe_dump(record, sort_keys=False))

    cmd = [
        sys.executable, "-m", "exp.dispatch_surface.run_precheck",
        "--arm-matrix", args.arm_matrix,
        "--task-suite", args.task_suite,
        "--servers", args.server,
        "--workers", "1",
        "--arms", arm,
        "--no-resume-filter",
        "--trials", "1",
        "--replan-steps", str(args.replan_steps),
        "--journal", str(unit_dir / "journal.jsonl"),
        "--per-step-out", str(unit_dir / "per_step.jsonl"),
        "--pool-record", str(record_path),
        "--pool-dir", str(pool_dir),
        "--gpus", "1",
    ]
    if args.conda_env:
        cmd += ["--conda-env", args.conda_env]
    subprocess.run(cmd, check=True)
    return unit_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-pass", choices=("compute", "latency"), required=True)
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--server", required=True, help="host:port (single replica launch)")
    ap.add_argument("--aprime-dir", required=True)
    ap.add_argument("--power-record", required=True,
                    help="frozen power-simulation record; R is read from it, "
                         "never from an operator argument (G2-B5)")
    ap.add_argument("--replan-steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--conda-env", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass-level attestation before any traffic.
    meta = _server_metadata(args.server)
    level = meta.get("monitor_level")
    expected = "SNAPSHOT" if args.bench_pass == "compute" else "OFF"
    if level != expected:
        raise SystemExit(
            f"pass {args.bench_pass} requires a server launched with "
            f"OPENPI_MONITOR_LEVEL={expected}, but metadata attests {level!r}"
        )
    if args.bench_pass == "compute":
        assert_compute_pass_metadata(meta)

    from exp.dispatch_surface.power_sim_cost_blocks import (
        record_digest as _power_digest,
        validate_power_record,
    )

    power = json.loads(pathlib.Path(args.power_record).read_text())
    blocks = validate_power_record(power)  # digest + constants + re-derived R
    args.blocks = blocks

    pools, block_mapping, pool_digest = materialize_block_pools(
        pathlib.Path(args.aprime_dir), out_dir / "pools", blocks, args.seed,
    )
    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    missing = set(CORE_ARMS) - set(matrix["arms"])
    if missing:
        raise SystemExit(f"arm matrix missing core arms: {sorted(missing)}")

    # Shared (block, arm-order) manifest — identical across the two passes.
    orders = {
        b: random.Random(args.seed * 100 + b).sample(list(CORE_ARMS), len(CORE_ARMS))
        for b in range(blocks)
    }
    aprime_h = hashlib.sha256()
    for f in sorted(pathlib.Path(args.aprime_dir).glob("*.init")):
        aprime_h.update(f.name.encode("utf-8"))
        aprime_h.update(f.read_bytes())
    manifest = {
        "bench_pass": args.bench_pass, "blocks": blocks, "seed": args.seed,
        "server": args.server, "monitor_level": level,
        "cuda_available": meta.get("cuda_available"),
        "gpu_name": meta.get("gpu_name"),
        "stage_devices": meta.get("stage_devices"),
        "stage_probe_backends": meta.get("stage_probe_backends"),
        "arm_orders": {str(b): orders[b] for b in orders},
        "policy_fingerprint": meta.get("policy_fingerprint"),
        "block_pool_digest": pool_digest,
        "block_init_mapping": block_mapping,
        "aprime_content_sha256": aprime_h.hexdigest(),
        "library_sha256": matrix.get("library_sha256"),
        "arm_matrix_sha256": hashlib.sha256(
            pathlib.Path(args.arm_matrix).read_bytes()
        ).hexdigest(),
        "power_record": str(args.power_record),
        "power_record_digest": _power_digest(power),
    }
    (out_dir / f"manifest_{args.bench_pass}.json").write_text(json.dumps(manifest, indent=2))

    # Warmup: one discarded episode per arm, before any measured window.
    warmup_pool = out_dir / "pools" / "warmup"
    for arm in CORE_ARMS:
        run_unit(args, arm, warmup_pool, tag=f"{args.bench_pass}_warmup_{arm}")
    if args.bench_pass == "compute":
        _ctrl(args.server, {"__ctrl__": "dump_metrics", "clear": True})  # drop warmup rows

    results = []
    for b in range(args.blocks):
        for arm in orders[b]:
            tag = f"{args.bench_pass}_b{b:02d}_{arm}"
            if args.bench_pass == "compute":
                _ctrl(args.server, {"__ctrl__": "dump_metrics", "clear": True})
            unit_dir = run_unit(args, arm, pools[b], tag=tag)
            unit = {"block": b, "arm": arm, "tag": tag,
                    "per_step_path": str(unit_dir / "per_step.jsonl")}
            if args.bench_pass == "compute":
                dump = _ctrl(args.server, {"__ctrl__": "dump_metrics", "clear": True})
                if dump.get("level") != "SNAPSHOT":
                    raise SystemExit(f"unit {tag}: dump level {dump.get('level')!r} != SNAPSHOT")
                unit["timing_rows"] = dump.get("timing", [])
            results.append(unit)

    out_path = out_dir / f"raw_{args.bench_pass}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"{len(results)} units -> {out_path}")


if __name__ == "__main__":
    main()
