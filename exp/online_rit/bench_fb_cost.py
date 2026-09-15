"""Measure a complete cost ledger for the online RIT line on one platform and mode.

Nothing is copied from another ledger. The first online RIT version uses
``--mode eager``; other modes are refused. On the serving hardware this measures:

* ``stage1_ms``, ``stage2_ms`` (language model only), the full head loop and
  the resumed loop at 1 / 2 / 4 remaining steps, priced directly from
  ``stage3_ladder_ms`` (the head/step least-squares fit is diagnostic only);
* ``fb_batch_ms[b]`` for b = 1 / 2 / 3: the **whole** side-evaluation path the
  interceptor pays -- the batched ``first_step_updates`` step plus the CPU
  transfer, the stored-update difference and the disagreement reduction
  (``feedback_from_updates``) -- with the GPU step alone in ``fb_gpu_ms[b]``;
* captured warm loops and the executed CPU feedback path;
* actual retrieval/gate/judge dispatch, frozen/learning feedback commits
  with persistent logs, and full snapshots over cold/partial/full windows.
  Host prices use the largest observed sample as a conservative allowance.

Output: one json in the ``CostLedger`` schema with ``fb_batch_ms`` and a
``provenance`` block (gpu, torch, mode, iterations, checkpoint identity,
scales sha). Runs in the GR00T island venv::

  python -m exp.online_rit.bench_fb_cost --checkpoint <ckpt> --scales <update_scales.npz> \\
      --library-pkl <library.pkl> --template-yaml <template.yaml> --knots <knots.json> \\
      --gate-theta <theta> \\
      --out exp/online_rit/data/cost/4090_eager/cost.json --iters 200 --mode eager
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
import time

import numpy as np
import torch

from exp.libero_groot.bench_profile import build_input, load_policy
from exp.online_rit.common import ALPHA, DENOISING_STEPS, H_EXEC, N_MIN, SNAPSHOT_EVERY, TIER_TS, WINDOW, canonical_sha256, sha256_file
from openpi.cache.components.online_rit import ContinuationSpec, feedback_from_updates, load_update_scales, tier_specs
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.types import groot_n15_schedule

SUPPORTED_MODES = ("eager",)


def _timed(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def fit_step_ladder(points: dict[int, float]) -> tuple[float, float]:
    """(head_ms, step_ms) by least squares over ``{steps: ms}``; head clamped at 0."""
    xs = np.array(sorted(points), dtype=np.float64)
    ys = np.array([points[int(k)] for k in xs], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    return max(float(intercept), 0.0), float(slope)


def measure_host_paths(components, stage1, spec, knots, library_sha, log_root, *, iters, task_key, theta):
    """Measure production dispatch, feedback commits and snapshots on real storage.

    The reported host prices are conservative maxima across observed cold,
    partially filled and full-window samples. Every dispatch includes the real
    key builder, retrieval and hysteresis gate; every commit uses the real judge
    and registry with file logging enabled. Setup and state seeding are untimed.
    """
    from openpi.cache.components.gate import ScoreHysteresisGate
    from openpi.cache.components.online_rit import ContinuationFeedback, OnlineRiskCurves, OnlineRitJudge
    from openpi.cache.online_state import CurveRegistry
    from openpi.cache.orchestrator import CacheOrchestrator
    from openpi.cache.types import CheckpointID

    dispatch, commit, snapshots, detail = {}, {}, {}, {}
    cp = CheckpointID.CP1

    def elapsed(fn):
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000

    for mode in ("threshold", "frozen", "learning"):
        for fill in ((0,) if mode == "threshold" else (0, WINDOW // 2, WINDOW)):
            registry = CurveRegistry(state_log_root=str(log_root))
            name = f"bench_{mode}_{fill}"
            curves = OnlineRiskCurves(knots=knots, tier_indices=[t.index for t in spec.tiers], alpha=ALPHA, window=WINDOW, n_min=N_MIN)
            for k in knots:
                for n in range(fill):
                    curves.update_batch(k, [ContinuationFeedback(t.index, 0.2, "shadow") for t in spec.tiers], ("seed", k, n))
            curves.update_enabled = mode == "learning"
            key = registry.attach(yaml_id=name, library_sha256=library_sha, fingerprint=name, factory=lambda: curves,
                                  snapshot_every=10**12)  # periodic writes measured separately
            judge = OnlineRitJudge(registry=registry, registry_key=key, spec=spec, delta=0.3, yaml_id=name)
            gate = ScoreHysteresisGate(theta_low=theta, theta_high=theta, j=3, probe_interval=3, L=6, include_ws=True)
            orch = CacheOrchestrator(
                storage=components["storage"], key_builder=components["key_builder"],
                gates={cp: gate}, judges={cp: components["judges"][cp] if mode == "threshold" else judge},
                search_strategies=components["search_strategies"], timer=components["timer"],
                write_policy=components["write_policy"], library_stats=components["library_stats"],
            )
            orch.on_episode_start(task_key=task_key)
            for n_fb in ((0,) if mode == "threshold" else (0, 1, 3)):
                feedback = [ContinuationFeedback(t.index, 0.25, "executed" if i == 0 else "shadow") for i, t in enumerate(spec.tiers[:n_fb])]
                ds, cs, ss = [], [], []
                for n in range(iters):
                    # Reset the gate outside timing to measure the full search path,
                    # a conservative bound also used on skip/no-candidate events.
                    gate.on_episode_start(task_key)
                    judge.on_episode_start(extra_metadata={"task_uid": f"{name}:{n_fb}:{n}", "attempt": 0})
                    result = []
                    ds.append(elapsed(lambda: result.append(orch.check(cp, stage1=stage1))))
                    if mode != "threshold":
                        if judge.pending_snapshot is None:
                            raise RuntimeError("host benchmark needs a retrieved candidate; check task/template/library")
                        cs.append(elapsed(lambda: judge.record_continuation(cp, judge.pending_snapshot, feedback)))
                        ss.append(elapsed(lambda: registry.flush(key, "bench")))
                    orch.clear()
                label = "threshold" if mode == "threshold" else "online"
                dispatch[label] = max(dispatch.get(label, 0.0), max(ds))
                if cs:
                    commit[f"{mode}:{n_fb}"] = max(commit.get(f"{mode}:{n_fb}", 0.0), max(cs))
                    snapshots[mode] = max(snapshots.get(mode, 0.0), max(ss))
                detail[f"{mode}:fill{fill}:fb{n_fb}"] = {"dispatch_ms": ds, "commit_ms": cs, "snapshot_ms": ss}
    return {"dispatch_ms": dispatch, "commit_ms": commit, "snapshot_ms": snapshots, "host_samples": detail}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scales", required=True, help="update_scales.npz (the feedback path scales the difference with it)")
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--template-yaml", required=True)
    ap.add_argument("--knots", required=True)
    ap.add_argument("--gate-theta", required=True, type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--denoising-steps", type=int, default=DENOISING_STEPS)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--mode", default="eager", help="execution mode actually run; only 'eager' is supported")
    args = ap.parse_args()
    if args.mode not in SUPPORTED_MODES:
        raise SystemExit(f"--mode {args.mode!r} is not supported by the serving path this bench measures; supported: {SUPPORTED_MODES}")
    if args.iters < 1 or args.warmup < 0:
        raise SystemExit("iters must be positive and warmup nonnegative")

    schedule = groot_n15_schedule(args.denoising_steps)
    from exp.rit_loto.build_loto_table import load_library
    library = load_library(args.library_pkl, expected_schedule=schedule, warm_ts=TIER_TS)
    payload = library.entries[0].payload
    task_key = str(payload.task_key)
    prompt = task_key.replace("_", " ")
    checkpoint = pathlib.Path(args.checkpoint)
    policy = load_policy(checkpoint, device="cuda", denoising_steps=args.denoising_steps)
    runner = GrootStagedRunner(policy.model)
    normalized = build_input(policy, checkpoint, prompt)
    tiers = tier_specs(list(TIER_TS), schedule)
    scales, masks, meta, scales_sha = load_update_scales(args.scales, [t.index for t in tiers])
    spec = ContinuationSpec(tiers=tiers, scales=scales, masks=masks, h_exec=H_EXEC, feedback_mode="fm1", schedule=schedule)
    if meta.get("library_sha256") != sha256_file(args.library_pkl):
        raise SystemExit("benchmark scales and library differ")

    with runner.session():
        stage1 = runner.run_stage1(normalized)
        stage2 = runner.run_stage2_llm(stage1)
        full = runner.run_stage3(stage2)
        snaps = payload.intermediates

        stage1_ms = _timed(lambda: runner.run_stage1(normalized), args.iters, args.warmup)
        stage2_ms = _timed(lambda: runner.run_stage2_llm(stage1), args.iters, args.warmup)
        ladder = {schedule.num_steps: _timed(lambda: runner.run_stage3(stage2), args.iters, args.warmup)}
        for t in TIER_TS:
            rem = schedule.remaining_steps(t)
            ladder[rem] = _timed(lambda t=t: runner.run_stage3_from(stage2, snaps[t], t, schedule=schedule), args.iters, args.warmup)
        captured, executed = {}, {}
        for t in TIER_TS:
            rem = schedule.remaining_steps(t)
            captured[rem] = max(ladder[rem], _timed(lambda t=t: runner.run_stage3_from(stage2, snaps[t], t, schedule=schedule, capture_first_step=True), args.iters, args.warmup))
            cap = runner.run_stage3_from(stage2, snaps[t], t, schedule=schedule, capture_first_step=True)
            executed[rem] = _timed(lambda t=t, cap=cap: feedback_from_updates(spec, payload, schedule, executed=(t, cap.first_step_input, cap.first_step_x), side=[]), args.iters, args.warmup)
        fb_gpu: dict[str, float] = {}
        fb_full: dict[str, float] = {}
        for batch in (1, 2, 3):
            fb_gpu[str(batch)] = 0.0
            fb_full[str(batch)] = 0.0

            def whole(items):
                pairs = runner.first_step_updates(stage2, items, schedule=schedule)
                side = [(t, x_in, x_out) for (t, _), (x_in, x_out) in zip(items, pairs)]
                fb, reasons = feedback_from_updates(spec, payload, schedule, executed=None, side=side)
                if reasons:
                    raise RuntimeError(f"feedback rejected during the bench: {reasons}")
                return fb

            for selected in itertools.combinations(TIER_TS, batch):
                items = [(t, snaps[t]) for t in selected]
                fb_gpu[str(batch)] = max(fb_gpu[str(batch)], _timed(lambda: runner.first_step_updates(stage2, items, schedule=schedule), args.iters, args.warmup))
                fb_full[str(batch)] = max(fb_full[str(batch)], _timed(lambda: whole(items), args.iters, args.warmup))
    from exp.rit_loto.build_loto_table import checkpoint_identity, load_template
    from openpi.cache.config import build_cache_components

    cfg, _ = load_template(args.template_yaml, args.library_pkl, pathlib.Path(args.out).parent / "host_bench")
    components = build_cache_components(cfg)
    host = measure_host_paths(components, stage1, spec, json.loads(pathlib.Path(args.knots).read_text())["knots"],
                              meta["library_sha256"], pathlib.Path(args.out).parent / "host_logs",
                              iters=args.iters, task_key=task_key, theta=args.gate_theta)
    head_ms, step_ms = fit_step_ladder(ladder)
    out = {
        "protocol": "online_rit_cost_v2",
        "teacher": "groot_libero",
        "schedule_id": schedule.schedule_id,
        "num_steps": schedule.num_steps,
        "stage1_ms": stage1_ms,
        "stage2_ms": stage2_ms,
        "stage3_head_ms": head_ms,
        "stage3_step_ms": step_ms,
        "stage3_full_loop_ms": ladder[schedule.num_steps],
        "stage3_ladder_ms": {str(k): v for k, v in sorted(ladder.items())},
        "captured_stage3_ms": captured,
        "executed_feedback_ms": executed,
        **host,
        "pricing": {"host": "maximum observed across cold/partial/full windows; full-search bound also on skips",
                    "snapshot": f"one per {SNAPSHOT_EVERY} learned batches plus at most one task-end write per episode",
                    "side": "maximum median across tier combinations at each batch size",
                    "stage3": "measured ladder consumed directly; linear fit diagnostic only"},
        "warm1_ms": stage1_ms + stage2_ms + ladder[1],
        "fb_batch_ms": fb_full,
        "fb_gpu_ms": fb_gpu,
        "linear_stage3": False,
        "provenance": (
            f"bench_fb_cost mode={args.mode} on {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
            f"{args.iters} iterations after {args.warmup} warmup, medians; every stage, the three warm tiers and "
            "the side-evaluation path (GPU step + CPU transfer + stored-update difference + disagreement "
            "reduction) measured in this run on one platform; stage3 head/step by least squares over the "
            "(1, 2, 4, 8)-step ladder."
        ),
        "bench": {
            "mode": args.mode, "iters": args.iters, "warmup": args.warmup, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "prompt": prompt, "head_dtype": str(full.action_pred.dtype), "checkpoint": str(checkpoint),
            "scales_sha256": scales_sha, "scales_library_sha256": meta.get("library_sha256"),
            "checkpoint_identity_sha256": checkpoint_identity(str(checkpoint))["sha256"],
            "template_sha256": sha256_file(args.template_yaml), "knots_sha256": sha256_file(args.knots),
            "code_sha256": canonical_sha256({str(p): sha256_file(p) for p in [pathlib.Path(__file__), pathlib.Path("src/openpi/cache/online_state.py"), pathlib.Path("src/openpi/cache/components/online_rit.py"), pathlib.Path("src/openpi/cache/groot/staged.py"), pathlib.Path("exp/online_rit/common.py")]}),
        },
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["sha256"] = sha256_file(path)
    print(f"wrote {path}: stage1 {stage1_ms:.2f} stage2 {stage2_ms:.2f} step {step_ms:.3f} fb_batch_ms={fb_full}")


if __name__ == "__main__":
    main()
