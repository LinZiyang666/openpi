"""Verification arm from the frozen LOTO fit, plus disjoint verify / smoke init pools.

``pools``  Sample the official A pool per task after excluding every init the
           150-episode shadow cohort used, draw ``verify`` + ``smoke`` inits that
           never overlap, materialise both pools (sorted by official index) and
           write one ``episode_filter`` JSON per pool so ``examples/libero/main.py``
           stamps the *official* index onto each episode instead of the subset
           position. The manifest carries every index, pool digest and the
           exclusion source's sha256.

``arm``    Read the frozen ``fits.json`` of the chosen ``fit_source``, invert the
           target inference ratio on that source's own score sample with the
           measured cost, cut the K=2 ladder, gate at the 0.85 score quantile of
           the same sample and write the arm yaml through the production loader.
           The record keeps delta / cuts / theta / dropped rungs / attainable
           range next to the deployed original arm's values for comparison.

Main venv. Usage:
  python -m exp.rit_loto.emit_verify_arm pools --suite libero_10 \
      --apool-dir exp/common/data/db_init/libero/libero_10_apool \
      --exclude-manifest exp/libero_groot/data/rit/shadow/libero_10/shadow_manifest.json \
      --out-dir exp/rit_loto/data/libero_10
  python -m exp.rit_loto.emit_verify_arm arm --suite libero_10 --fits exp/rit_loto/data/libero_10/fits.json \
      --fit-source loto_all --target-ir 70 --cost exp/libero_groot/config/rit/cost_groot_libero_measured.json \
      --template-yaml exp/libero_groot/config/rit/libero_10/template.yaml \
      --pool-manifest exp/rit_loto/data/libero_10/verify_pool_manifest.json \
      --original-arm-record exp/libero_groot/config/rit/libero_10/arm_record.json \
      --config-out exp/rit_loto/config/libero_10 --record-out exp/rit_loto/data/libero_10/frozen_run.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import yaml

from exp.dispatch_surface.emit_precheck_yamls import LAYER_SECONDARY
from exp.dispatch_surface.split_init_pools import OFFICIAL_PER_TASK, materialize_pool
from exp.libero_groot.emit_rit_arms import WARM_TS, _judge_from_cuts, build_arm, gate_theta, load_cost, write_arm
from exp.robocasa365 import rit_cost_rc as rc

from exp.rit_loto.build_loto_table import IDENTITY_KEYS, code_sha256, git_commit, require_code_identity, sha256_file
from exp.rit_loto.fit_loto import LEGACY_ALPHA, _cost_to_json, deserialize_fit

DEFAULT_SEED = 20260913
DEFAULT_PER_TASK_VERIFY = 5
DEFAULT_PER_TASK_SMOKE = 1
DEFAULT_TARGET_IR = 70.0
DEFAULT_K = 2
FROZEN_ALPHA = float(LEGACY_ALPHA)


# ------------------------------------------------------------------
# Pools
# ------------------------------------------------------------------


def _filter_entries(task_id: int, indices: list[int]) -> list[dict]:
    """``episode_filter`` rows: subset position (sorted order) -> official index."""
    return [{"task_id": int(task_id), "subset_init_state_idx": pos, "orig_init_state_idx": int(idx)}
            for pos, idx in enumerate(sorted(int(i) for i in indices))]


def sample_pools(suite: str, apool_dir: str | pathlib.Path, exclude_manifest: str | pathlib.Path,
                 per_task_verify: int, per_task_smoke: int, seed: int, out_dir: str | pathlib.Path) -> dict:
    """Disjoint verify / smoke inits per task from the A pool minus the shadow cohort."""
    apool_dir, out_dir = pathlib.Path(apool_dir), pathlib.Path(out_dir)
    manifest_path = pathlib.Path(exclude_manifest)
    excl = json.loads(manifest_path.read_text(encoding="utf-8"))
    if excl.get("suite") != suite:
        raise SystemExit(f"exclusion manifest is for {excl.get('suite')!r}, not {suite!r}")
    assignment = {int(t): info for t, info in excl["assignment"].items()}
    verify_assign, smoke_assign, exclude, verify_idx, smoke_idx = {}, {}, {}, {}, {}
    for tid in sorted(assignment):
        info = assignment[tid]
        used = {int(i) for i in info.get("fit", [])} | {int(i) for i in info.get("cal", [])}
        cands = sorted(set(range(OFFICIAL_PER_TASK)) - used)
        need = int(per_task_verify) + int(per_task_smoke)
        if len(cands) < need:
            raise SystemExit(f"task {tid}: {len(cands)} inits left after exclusion, need {need}")
        rng = np.random.default_rng([int(seed), int(tid)])
        picked = [cands[int(i)] for i in rng.choice(len(cands), size=need, replace=False)]
        verify = sorted(picked[: int(per_task_verify)])
        smoke = sorted(picked[int(per_task_verify):])
        if set(verify) & set(smoke) or (set(verify) | set(smoke)) & used:
            raise SystemExit(f"task {tid}: pools overlap")
        verify_assign[tid] = {"task_name": info["task_name"], "verify": verify}
        smoke_assign[tid] = {"task_name": info["task_name"], "smoke": smoke}
        exclude[str(tid)], verify_idx[str(tid)], smoke_idx[str(tid)] = sorted(used), verify, smoke
    verify_dir, smoke_dir = out_dir / "verify_pool", out_dir / "smoke_pool"
    for d in (verify_dir, smoke_dir):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"pool dir must be empty or absent: {d}")
    dig_v = materialize_pool(apool_dir, verify_dir, verify_assign, ["verify"])
    dig_s = materialize_pool(apool_dir, smoke_dir, smoke_assign, ["smoke"])
    filters = {"verify": [e for tid in sorted(verify_assign) for e in _filter_entries(tid, verify_assign[tid]["verify"])],
               "smoke": [e for tid in sorted(smoke_assign) for e in _filter_entries(tid, smoke_assign[tid]["smoke"])]}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, entries in filters.items():
        p = out_dir / f"{name}_filter.json"
        p.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
        paths[name] = str(p)
    manifest = {
        "protocol": "rit_loto_verify_pool_v1", "suite": suite, "seed": int(seed),
        "per_task": {"verify": int(per_task_verify), "smoke": int(per_task_smoke)},
        "apool_dir": str(apool_dir), "exclude_manifest": str(manifest_path),
        "exclude_manifest_sha256": sha256_file(manifest_path), "exclude": exclude,
        "task_names": {str(t): assignment[t]["task_name"] for t in sorted(assignment)},
        "verify": verify_idx, "smoke": smoke_idx,
        "pool_dirs": {"verify": str(verify_dir), "smoke": str(smoke_dir)},
        "pool_digests": {"verify": dig_v, "smoke": dig_s}, "filters": paths,
        "n_episodes": {"verify": sum(len(v) for v in verify_idx.values()), "smoke": sum(len(v) for v in smoke_idx.values())},
    }
    (out_dir / "verify_pool_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


# ------------------------------------------------------------------
# Arm and the frozen run record
# ------------------------------------------------------------------

FROZEN_PROTOCOL = "rit_loto_frozen_run_v1"
RUN_TAGS = {"smoke": "smoke", "verify": "verify"}
VERIFY_PROTOCOL = {"n_sample": 2000, "sample_seed": 20260914, "root_seed": 20260914,
                   "tol": 0.05, "n_boot": 1000, "seed": 0, "min_rows": 200, "min_episodes": 20,
                   "min_event_episodes": 5, "min_valid_fraction": 0.9}


def validate_pool_manifest(pool: dict) -> None:
    """Require the frozen LIBERO-10 50/10 disjoint pools outside the 150-episode shadow cohort."""
    if pool.get("suite") != "libero_10" or pool.get("per_task") != {"verify": 5, "smoke": 1} \
            or pool.get("n_episodes") != {"verify": 50, "smoke": 10}:
        raise SystemExit("pool manifest must describe libero_10: 50 verify episodes and 10 smoke episodes")
    for name, n in (("verify", 5), ("smoke", 1), ("exclude", 15)):
        groups = pool.get(name, {})
        if set(groups) != {str(t) for t in range(10)}:
            raise SystemExit(f"pool manifest {name} must contain exactly tasks 0..9")
        for t, indices in groups.items():
            if not isinstance(indices, list) or len(indices) != n or len(set(indices)) != n \
                    or any(type(i) is not int or not 0 <= i < 50 for i in indices) or indices != sorted(indices):
                raise SystemExit(f"pool manifest {name} task {t}: invalid official indices")
    for t in map(str, range(10)):
        v, s, e = (set(pool[name][t]) for name in ("verify", "smoke", "exclude"))
        if v & s or (v | s) & e:
            raise SystemExit(f"pool manifest task {t}: verify/smoke/shadow overlap")


def validate_frozen_record(rec: dict) -> None:
    """Validate the complete deployment contract and require the code that produced it."""
    fixed = {"protocol": FROZEN_PROTOCOL, "suite": "libero_10", "fit_source": "loto_all", "k": 2,
             "alpha": FROZEN_ALPHA, "target_ir": 70.0, "run_tags": RUN_TAGS,
             "pool_counts": {"verify": 50, "smoke": 10}, "pool_per_task": {"verify": 5, "smoke": 1},
             "verify_protocol": VERIFY_PROTOCOL, "compile_stage1": False}
    for key, value in fixed.items():
        if rec.get(key) != value:
            raise SystemExit(f"frozen record {key}: expected {value!r}, got {rec.get(key)!r}")
    if rec.get("unreachable") or not rec.get("arm"):
        raise SystemExit("frozen record has no emitted arm")
    for key in ("fits_sha256", "template_sha256", "pool_manifest_sha256", "cost"):
        if not rec.get(key):
            raise SystemExit(f"frozen record lacks {key}")
    ident = rec.get("identity", {})
    for key in IDENTITY_KEYS:
        if key not in ident or ident[key] in (None, "", "unknown", "missing"):
            raise SystemExit(f"frozen identity lacks {key}")
    if ident["suite"] != rec["suite"] or ident["template_sha256"] != rec["template_sha256"] \
            or ident["h_exec"] != 5 or ident["schedule_id"] != "groot_n15_k8_v1":
        raise SystemExit("frozen identity suite/template/H_exec/schedule mismatch")
    if rec["arm"].get("tiers") != ["full", "warm75"] or not rec["arm"].get("yaml_sha256"):
        raise SystemExit("frozen record lacks the emitted K=2 arm identity")
    validate_pool_manifest(rec.get("pool", {}))
    require_code_identity(ident["code_sha256"])
    require_code_identity(rec.get("code_sha256"))


def emit_arm(suite: str, fit_source: str, fits_path: str | pathlib.Path, cost: rc.StageCost, target_ir: float,
             template_path: str | pathlib.Path, config_out: str | pathlib.Path, *, pool_manifest: str | pathlib.Path,
             k: int = DEFAULT_K, original_arm_record: str | pathlib.Path | None = None) -> dict:
    """One K-tier arm addressed at ``target_ir`` from the frozen fit, plus the frozen run record.

    The returned record is what every later stage binds to: the fits digest and
    the table identity it carries (library / checkpoint / template / W / H_exec /
    schedule / code digests), the arm yaml digest, the pool manifest digest and
    the run tags. Refuses a template that is not the one the fits were built on,
    a fit source that is unavailable, an alpha other than the frozen one, or a
    pool manifest of another suite.
    """
    fits_path, template_path, pool_manifest = pathlib.Path(fits_path), pathlib.Path(template_path), pathlib.Path(pool_manifest)
    fits = json.loads(fits_path.read_text(encoding="utf-8"))
    if (suite, fit_source, k, float(target_ir)) != ("libero_10", "loto_all", 2, 70.0):
        raise SystemExit("verification is frozen to libero_10 / loto_all / K=2 / IR70")
    if fits.get("suite") != suite:
        raise SystemExit(f"fits.json is for {fits.get('suite')!r}, not {suite!r}")
    identity = fits.get("identity") or {}
    template_sha = sha256_file(template_path)
    if identity.get("template_sha256") != template_sha:
        raise SystemExit(f"template {template_path} (sha {template_sha[:12]}) is not the template the fits were built on "
                         f"({str(identity.get('template_sha256'))[:12]})")
    require_code_identity(identity.get("code_sha256"))
    if fits.get("parity_gate_status") != "PASS" or fits.get("cost") != _cost_to_json(cost):
        raise SystemExit("fits must carry a PASS parity gate and the same measured cost")
    block = fits.get(fit_source)
    if block is None or block.get("fit_unavailable"):
        raise SystemExit(f"fit source {fit_source!r} is unavailable in {fits_path}: {None if block is None else block.get('reason')}")
    if float(block.get("alpha", -1)) != FROZEN_ALPHA:
        raise SystemExit(f"fit source {fit_source!r} alpha {block.get('alpha')} != frozen {FROZEN_ALPHA}")
    rec = block["fits"].get(str(k))
    if rec is None:
        raise SystemExit(f"fit source {fit_source!r} has no K={k} fit")
    pool = json.loads(pool_manifest.read_text(encoding="utf-8"))
    if pool.get("suite") != suite:
        raise SystemExit(f"pool manifest is for {pool.get('suite')!r}, not {suite!r}")
    validate_pool_manifest(pool)
    warm_ts = tuple(WARM_TS[: k - 1])
    fit = deserialize_fit(rec, cost, WARM_TS)
    if fit.alpha != FROZEN_ALPHA or [t.y_key for t in fit.tiers] != ["y_full", "y_rem2"]:
        raise SystemExit("K=2 fit alpha or label contract differs from the frozen protocol")
    s = np.asarray(block["s_sample"], dtype=np.float64)
    lo, hi = rc.attainable_range(fit, s, cost)
    record: dict = {
        "protocol": FROZEN_PROTOCOL, "suite": suite, "fit_source": fit_source, "k": int(k), "alpha": FROZEN_ALPHA,
        "target_ir": float(target_ir), "fits_json": str(fits_path), "fits_sha256": sha256_file(fits_path),
        "identity": identity, "template_sha256": template_sha, "pool_manifest": str(pool_manifest),
        "pool_manifest_sha256": sha256_file(pool_manifest), "pool_counts": dict(pool.get("n_episodes", {})),
        "pool_per_task": dict(pool.get("per_task", {})), "run_tags": dict(RUN_TAGS),
        "pool": pool, "cost": _cost_to_json(cost), "verify_protocol": dict(VERIFY_PROTOCOL),
        "compile_stage1": False,
        "attainable_ir_range": [float(lo), float(hi)], "n_score_sample": int(s.size),
        "code_sha256": code_sha256(), "git_commit": git_commit(),
    }
    if not (lo <= float(target_ir) <= hi):
        record["unreachable"] = True
        return record
    sol = rc.delta_for_ir(fit, s, float(target_ir), cost)
    cuts = rc.cuts_for(fit, float(sol["delta"]))
    judge, dropped = _judge_from_cuts(cuts, warm_ts)
    theta = gate_theta(s)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    doc = build_arm(template, judge, layer=LAYER_SECONDARY, theta=theta)
    arm_name = f"loto_k{k}_ir{int(round(float(target_ir)))}"
    path = pathlib.Path(config_out) / f"{arm_name}.yaml"
    sha = write_arm(doc, path)
    record["arm"] = {
        "name": arm_name, "yaml": str(path), "yaml_sha256": sha, "delta": float(sol["delta"]),
        "predicted_ir": float(sol["predicted_ir"]), "ir_gap": float(sol["ir_gap"]),
        "cuts": [None if not np.isfinite(c) else float(c) for c in cuts], "tiers": [t.name for t in fit.tiers],
        "dropped_rungs": dropped, "gate_theta": float(theta), "judge": judge,
    }
    if original_arm_record:
        orig = json.loads(pathlib.Path(original_arm_record).read_text(encoding="utf-8"))
        name = next((n for n, a in orig["arms"].items() if a.get("rule") == "rit" and int(a.get("k", 0)) == k
                     and int(round(float(a.get("target_ir", -1)))) == int(round(float(target_ir)))), None)
        if name is not None:
            a = orig["arms"][name]
            record["original_arm"] = {"arm": name, "delta": a["delta"], "cuts": a["cuts"], "gate_theta": orig["gate_theta"],
                                      "predicted_ir": a["predicted_ir"], "dropped_rungs": a.get("dropped_rungs", []),
                                      "cost_provenance": orig.get("cost", {}).get("provenance", "")[:80]}
    validate_frozen_record(record)
    return record


def load_frozen_record(path: str | pathlib.Path) -> tuple[dict, str]:
    """``(record, sha256 of its bytes)``; refuses anything that is not a complete frozen run record with an arm."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise SystemExit(f"frozen run record missing: {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("protocol") != FROZEN_PROTOCOL:
        raise SystemExit(f"{path}: not a {FROZEN_PROTOCOL} record")
    if rec.get("unreachable") or "arm" not in rec:
        raise SystemExit(f"{path}: the record carries no emitted arm (target unreachable?)")
    validate_frozen_record(rec)
    return rec, sha256_file(path)


def main() -> None:
    """CLI: ``pools`` (verify / smoke inits + filters + manifest) and ``arm`` (arm yaml + frozen run record)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pools")
    p.add_argument("--suite", required=True)
    p.add_argument("--apool-dir", required=True)
    p.add_argument("--exclude-manifest", required=True)
    p.add_argument("--per-task-verify", type=int, default=DEFAULT_PER_TASK_VERIFY)
    p.add_argument("--per-task-smoke", type=int, default=DEFAULT_PER_TASK_SMOKE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--suite", required=True)
    a.add_argument("--fits", required=True)
    a.add_argument("--fit-source", default="loto_all")
    a.add_argument("--k", type=int, default=DEFAULT_K)
    a.add_argument("--target-ir", type=float, default=DEFAULT_TARGET_IR)
    a.add_argument("--cost", required=True)
    a.add_argument("--template-yaml", required=True)
    a.add_argument("--pool-manifest", required=True)
    a.add_argument("--original-arm-record", default="")
    a.add_argument("--config-out", required=True)
    a.add_argument("--record-out", required=True)
    args = ap.parse_args()
    if args.cmd == "pools":
        m = sample_pools(args.suite, args.apool_dir, args.exclude_manifest, args.per_task_verify, args.per_task_smoke,
                         args.seed, args.out_dir)
        print(f"pools -> {args.out_dir}: verify {m['n_episodes']['verify']} / smoke {m['n_episodes']['smoke']} episodes")
        return
    cost = load_cost(pathlib.Path(args.cost))
    rec = emit_arm(args.suite, args.fit_source, args.fits, cost, args.target_ir, args.template_yaml, args.config_out,
                   pool_manifest=args.pool_manifest, k=args.k, original_arm_record=args.original_arm_record or None)
    out = pathlib.Path(args.record_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    if rec.get("unreachable"):
        raise SystemExit(f"target IR {args.target_ir} outside attainable {rec['attainable_ir_range']}; record written, no arm")
    arm = rec["arm"]
    print(f"arm {arm['name']}: delta {arm['delta']:.4f} cuts {arm['cuts']} theta {arm['gate_theta']:.6f} "
          f"predicted IR {arm['predicted_ir']:.2f} dropped {arm['dropped_rungs']} -> {arm['yaml']}; frozen record -> {out}")


if __name__ == "__main__":
    main()
