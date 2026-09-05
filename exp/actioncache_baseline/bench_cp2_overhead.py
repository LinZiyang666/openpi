"""Measure the per-decision CP2 cache overhead on the real orchestrator (plan §3.10).

``exp/cache_latency_bench`` drives CP1 without a model and without the D2H
of the key build, so it cannot exercise the CP2 builder. This harness loads
the real model, replays a teacher cohort's H5 decisions (Stage 1 rebuilt from
the stored tensors, ``run_stage2_capture`` on the GPU) and then times the
production ``CacheOrchestrator.check(CP2, stage2=...)`` assembled from the
arm YAML by ``build_cache_components`` — sparse projection on the GPU tensor,
the single D2H of the key, the suite-wide filter + cosine search over the
whole library and the affine normalisation + threshold judge — with
``torch.cuda.synchronize()`` on both sides of the call.

Timer contract: the arm YAML ships ``timer.enabled: false`` (production has
no probes); the harness forces ``timer.enabled = True`` and a monitor level
that lets ``SystemTimer`` record, and reads the orchestrator's own probes
(``cp2_collect``, ``cp2_gate``, ``cp2_build``, ``cp2_search``, ``cp2_judge``,
``cp2_fetch``) per decision via ``on_task_begin`` / ``summary(task_only)``.
A run whose core segments recorded nothing is a harness fault and aborts.

Output contract: ``per_decision.csv`` (``episode, step_idx, total_ms`` + one
``<segment>_ms`` column per probe) and ``overhead.json`` with ``{suite,
library, library_sha256, model, config_digest, hardware, n_decisions, cold
{median, p95}, warm {median, p95}, per_segment {name: {median, p95, count}},
verdict}``. Acceptance / fallback thresholds of the plan (warm total P95
<= 10 ms / 10-40 ms / > 40 ms) are reported as ``verdict``; above 40 ms the
next step is a per-segment profile (``halt_profile_segments``), never a
backend change decided without one.

Usage:
  uv run python -m exp.actioncache_baseline.bench_cp2_overhead \\
      --suite libero_spatial --cache-yaml <arm yaml> --cohort-h5-root <dir> \\
      --config-name pi05_libero --checkpoint-dir <ckpt> --out-dir <dir> \\
      [--max-decisions 2000] [--cold 50]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import platform
import subprocess
import time
from typing import Iterable

import numpy as np
import torch

from exp.actioncache_baseline import libs
from openpi.cache.config import build_cache_components, load_cache_config
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.types import CheckpointID

SEGMENTS = ("cp2_collect", "cp2_gate", "cp2_build", "cp2_search", "cp2_judge", "cp2_fetch")
#: Probes every decision must record (fetch only fires on hits).
CORE_SEGMENTS = ("cp2_collect", "cp2_build", "cp2_search", "cp2_judge")
VERDICT_OK_MS = 10.0
VERDICT_HALT_MS = 40.0


def hardware_info(device: torch.device) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": str(device), "host": platform.node()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip().splitlines()[0]
        except Exception:  # noqa: BLE001 - provenance only
            info["driver"] = "unknown"
    return info


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pctl(xs: list[float]) -> dict:
    if not xs:
        return {"median": None, "p95": None, "count": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "count": int(a.size)}


def verdict_for(warm_p95_ms: float | None) -> str:
    """Plan §3.10 acceptance / fallback rule on the warm total P95."""
    if warm_p95_ms is None:
        return "insufficient_decisions"
    if warm_p95_ms <= VERDICT_OK_MS:
        return "ok_report"
    if warm_p95_ms <= VERDICT_HALT_MS:
        return "report_with_caption"
    return "halt_profile_segments"


def build_orchestrator(cache_yaml: str | pathlib.Path):
    """Real components from the arm YAML with the timer forced on.

    Returns ``(config, components, orchestrator, timer)``. The monitor level is
    raised to BASIC so ``SystemTimer`` records (the process default OFF would
    silently turn every ``measure()`` into a no-op).
    """
    from openpi.serving import monitor as _monitor

    if _monitor.get_monitor_level() < _monitor.MonitorLevel.BASIC:
        _monitor.set_monitor_level(_monitor.MonitorLevel.BASIC)
    config = load_cache_config(cache_yaml)
    if "cp2" not in config.checkpoints:
        raise SystemExit("--cache-yaml must be a CP2 arm config")
    config.timer.enabled = True
    components = build_cache_components(config)
    timer = components["timer"]
    if not getattr(timer, "_enabled", False):
        raise SystemExit("SystemTimer stayed disabled after forcing timer.enabled; cannot measure segments")
    orch = CacheOrchestrator(
        storage=components["storage"], key_builder=components["key_builder"],
        gates=components["gates"], judges=components["judges"],
        search_strategies=components["search_strategies"], timer=timer,
        write_policy=components.get("write_policy"),
        offline_writers=components.get("offline_writers", ()),
        library_stats=components.get("library_stats"),
    )
    return config, components, orch, timer


def run_decisions(orch: CacheOrchestrator, timer, decisions: Iterable[tuple[str, int, object]], *,
                  device: torch.device, out_dir: str | pathlib.Path, cold: int, max_decisions: int) -> dict:
    """Time ``check(CP2)`` per decision and write ``per_decision.csv``.

    ``decisions`` yields ``(episode, step_idx, stage2)`` with ``stage2`` already
    on ``device``; the generator may lazily run the model. Returns totals and
    per-segment samples.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals: list[float] = []
    seg_samples: dict[str, list[float]] = {s: [] for s in SEGMENTS}
    rows: list[list] = []
    current_episode = None
    n = 0
    with torch.no_grad():
        for episode, step_idx, stage2 in decisions:
            if n >= max_decisions:
                break
            if episode != current_episode:
                if current_episode is not None:
                    orch.on_episode_end()
                orch.on_episode_start(task_key=str(episode), episode_id=str(episode))
                current_episode = episode
            timer.on_task_begin()  # isolate this decision's probe records
            _sync(device)
            t0 = time.perf_counter()
            orch.check(CheckpointID.CP2, stage2=stage2)
            _sync(device)
            total_ms = (time.perf_counter() - t0) * 1000.0
            orch.clear()
            per = timer.summary(task_only=True)
            row = [episode, step_idx, total_ms]
            for s in SEGMENTS:
                st = per.get(s)
                val = float(st.mean_ms) if st is not None else None
                if val is not None:
                    seg_samples[s].append(val)
                row.append(val)
            missing = [s for s in CORE_SEGMENTS if per.get(s) is None]
            if missing:
                raise SystemExit(f"decision {episode}:{step_idx} recorded no {missing} probe — timer not wired")
            totals.append(total_ms)
            rows.append(row)
            n += 1
    if current_episode is not None:
        orch.on_episode_end()
    with (out_dir / "per_decision.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "step_idx", "total_ms", *[f"{s}_ms" for s in SEGMENTS]])
        w.writerows(rows)
    cold_t, warm_t = totals[:cold], totals[cold:]
    return {
        "n_decisions": n, "cold_decisions": len(cold_t),
        "cold": _pctl(cold_t), "warm": _pctl(warm_t),
        "per_segment": {s: _pctl(v) for s, v in seg_samples.items()},
        "verdict": verdict_for(_pctl(warm_t)["p95"]),
    }


def write_record(out_dir: str | pathlib.Path, *, suite: str, cache_yaml: str | pathlib.Path, config,
                 components: dict, device: torch.device, cohort_root: str | None, model_binding: dict | None,
                 measured: dict) -> dict:
    meta = components["storage"].artifact_meta or {}
    record = {
        "protocol": libs.PROTOCOL,
        "suite": suite,
        "cache_yaml": str(pathlib.Path(cache_yaml).resolve()),
        "config_digest": hashlib.sha256(pathlib.Path(cache_yaml).read_bytes()).hexdigest(),
        "library": config.backend.in_memory.preload_path,
        "library_sha256": meta.get("library_sha256"),
        "library_entries": meta.get("entry_count"),
        "projection": meta.get("projection"),
        "model": {"library_model": meta.get("model"), "bound": model_binding},
        "hardware": hardware_info(device),
        "cohort_root": None if cohort_root is None else str(pathlib.Path(cohort_root).resolve()),
        **measured,
        "thresholds_ms": {"ok": VERDICT_OK_MS, "halt": VERDICT_HALT_MS},
        "timer_enabled": bool(getattr(components["timer"], "_enabled", False)),
        "git_commit": libs.git_commit(),
    }
    libs.dump_json(pathlib.Path(out_dir) / "overhead.json", record)
    return record


def _cohort_decisions(h5_root: str, model, tokenizer, device: torch.device):
    """Lazily replay every step of every cohort H5 through ``run_stage2_capture``."""
    import h5py

    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
        _self_check_tokenizer_consistency,
    )

    h5_files = sorted(pathlib.Path(h5_root).rglob("*.h5"))
    if not h5_files:
        raise SystemExit(f"no H5 under {h5_root}")
    for i, h5_path in enumerate(h5_files):
        if i == 0:
            _self_check_tokenizer_consistency(h5_path, model, tokenizer, device)
        with h5py.File(h5_path, "r") as f:
            task = str(f.attrs.get("task", ""))
            episode = h5_path.stem
            for step_idx, group in libs.iter_steps(f):
                fake = _build_fake_stage1_with_masks(
                    group, task_str=task, tokenizer=tokenizer, model=model, device=device,
                )
                yield episode, step_idx, model.run_stage2_capture(fake)


def bench(args: argparse.Namespace) -> dict:
    from exp.common.build_in_memory_cache_artifact import _load_pi05_for_llm_extract

    device = torch.device(args.device)
    config, components, orch, timer = build_orchestrator(args.cache_yaml)
    meta = components["storage"].artifact_meta or {}
    # The library keys, the shadow cuts and this run must all come from one
    # set of weights (plan §3.7 binding; full-content digest).
    model_binding = libs.assert_model_binding(meta.get("model"), args.checkpoint_dir)
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)
    model.eval()
    orch.on_task_begin()
    measured = run_decisions(
        orch, timer, _cohort_decisions(args.cohort_h5_root, model, tokenizer, device),
        device=device, out_dir=args.out_dir, cold=args.cold, max_decisions=args.max_decisions,
    )
    orch.on_task_end()
    return write_record(args.out_dir, suite=args.suite, cache_yaml=args.cache_yaml, config=config,
                        components=components, device=device, cohort_root=args.cohort_h5_root,
                        model_binding=model_binding, measured=measured)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(libs.SUITE_TAGS))
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--cohort-h5-root", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--cold", type=int, default=50, help="first N decisions reported as cold")
    args = ap.parse_args()
    rec = bench(args)
    print(json.dumps({k: rec[k] for k in ("n_decisions", "cold", "warm", "verdict")}))


if __name__ == "__main__":
    main()
