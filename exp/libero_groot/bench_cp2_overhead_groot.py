"""GR00T CP2 preflight (plan §3.11): the encoder cost ``E`` and the real per-decision overhead.

Two measurements, both required before a single GR00T CP2 arm is emitted:

``encoder-cost`` / ``certify-encoder``
    The extra model forward the CP2 key needs -- ``run_cp2_key_source``, i.e.
    the action head's ``process_backbone_output`` (as its tensor-only twin,
    asserted bitwise equal to the production path) + ``state_encoder`` run once
    more per decision. Measured under the owner's teacher-cost protocol: the
    RTX 4090, bf16 autocast, the representative prompt shape (N = 566 tokens),
    ``torch.compile(mode="reduce-overhead")``, 30 warmup + 200 timed calls,
    median of ``torch.cuda.synchronize()``-bracketed wall clock. The
    measurement loop sits in a ``cudaProfilerApi`` range with an NVTX cell
    marker; the record is written *uncertified* and ``certify-encoder`` folds
    the external Nsight trace in (cudaGraphLaunch count == iters x graphs per
    call, zero captures after warmup). An eager or uncertified number is never
    labelled CUDA-Graph. Output: ``cost_groot_cp2_encoded_<suite>.json`` with
    ``cp2_key_encoder_ms`` and the sha256 of the teacher cost table it
    accompanies (``libs.groot_cost_record`` binds the two).

``overhead``
    The real decision overhead on the served orchestrator: cohort HDF5 steps
    (accepted manifest only) are rebuilt and run through stage 2 *outside* the
    timed boundary; the boundary is then
    ``synchronize -> run_cp2_key_source (session, cp2_encode probe) -> leave
    session -> check(CP2) -> synchronize``. Per decision the CSV carries
    ``total_ms``, ``cp2_encode_ms``, ``check_total_ms`` and the orchestrator's
    ``cp2_collect / gate / build / search / judge / fetch`` probes;
    ``cp2_build`` holds only pad/concat/projection/D2H. The verdict follows the
    plan's warm-total P95 rule (<= 10 ms report / 10-40 ms caption / > 40 ms
    stop); the exporter refuses to emit without an ``ok_report`` /
    ``report_with_caption`` record. ``cp2_encode_ms`` here is diagnostic; the
    analytic ``E`` enters IR through the certified record above.

Usage:
  python -m exp.libero_groot.bench_cp2_overhead_groot --mode encoder-cost --suite libero_spatial \\
      --checkpoint <ckpt> --cost-record exp/libero_groot/config/rit/cost_groot_libero_measured.json \\
      --out exp/libero_groot/config/actioncache/cost_groot_cp2_encoded_libero_spatial.json
  python -m exp.libero_groot.bench_cp2_overhead_groot --mode certify-encoder --out <that json> --cuda-trace <csv>
  python -m exp.libero_groot.bench_cp2_overhead_groot --mode overhead --suite libero_spatial \\
      --checkpoint <ckpt> --library-pkl <cp2.pkl> --accepted-manifest <accepted_shadow_manifest.json> \\
      --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import pathlib
import platform
import statistics
import time
from typing import Any

import h5py
import torch
import yaml

from openpi.cache.groot.cp2_key_builder import KEY_BUILDER_TYPE
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.types import CheckpointID

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.bench_cp2_overhead import (
    SEGMENTS,
    _pctl,
    build_orchestrator,
    hardware_info,
    verdict_for,
)
from exp.actioncache_baseline.export_arms import cp2_arm_yaml
from exp.libero_groot.cp2_reconstruct import (
    STAGE1_PATH,
    TemplateCache,
    build_template,
    h5_task,
    load_groot_libero_policy,
    reconstruct_stage1,
)

PROFILE = libs.GROOT_LIBERO
COMPILE_MODE = "reduce-overhead"
#: The representative prompt of the owner's G-M cell (N = 566 tokens on LIBERO).
REPRESENTATIVE_PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"
REPRESENTATIVE_N = 566
UNCERTIFIED = "uncertified: no CUDA trace parsed"
CORE_SEGMENTS_GROOT = ("cp2_encode", "cp2_collect", "cp2_build", "cp2_search", "cp2_judge")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def encoder_trace_marker(record: dict) -> str:
    """The NVTX range name of one encoder-cost measurement, a digest of the record's identity fields."""
    fields = {k: record.get(k) for k in ("suite", "schedule_id", "prompt_sha256", "mode", "warmup", "iters",
                                          "expected_cudagraph_launch_count", "ckpt_weights_digest", "gpu_uuid", "ts")}
    identity = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"openpi_acbE_{_sha256_text(identity)[:24]}"


def _timed(fn) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.monotonic()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.monotonic() - t0) * 1e3


# ------------------------------------------------------------------
# encoder cost E
# ------------------------------------------------------------------


def measure_encoder_cost(args: argparse.Namespace) -> dict:
    """Time the CP2 key-source encoders under CUDA-Graph (``E``), writing an *uncertified* record.

    The record carries every identity the consumer binds (G2-B4): suite,
    teacher table sha and calibration checkpoint, this suite's checkpoint
    (``ckpt_sha256`` and the library's ``weights_digest``), GPU name / UUID, the sampling (N tokens,
    warmup, iterations, mode) and the expected graph-launch count. Sampling
    other than the frozen ``libs.GROOT_E_SAMPLING`` is allowed for debugging
    only and is stamped ``debug_sampling``; ``certify_encoder`` refuses it.
    """
    from exp.robocasa365.bench_groot_stages import (
        checkpoint_identity,
        cuda_profiler_range,
        cudagraph_skips,
        gpu_provenance,
        inductor_counters,
        nvtx_measurement_range,
        unique_graph_count,
    )

    cost_path = pathlib.Path(args.cost_record)
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    if cost.get("teacher") != PROFILE.name or cost.get("schedule_id") != PROFILE.denoise_schedule:
        raise SystemExit(f"{cost_path} is not the {PROFILE.name} teacher cost record")
    if cost.get("mode") != COMPILE_MODE or not cost.get("certified"):
        raise SystemExit(f"{cost_path} must be a certified {COMPILE_MODE} record")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("E must be measured on the CUDA device of the teacher table")
    model_identity = libs.weights_digest(args.checkpoint)
    ckpt_sha256 = checkpoint_identity(pathlib.Path(args.checkpoint))
    provenance = gpu_provenance(device.index or 0)
    if provenance["gpu_uuid"] != cost.get("gpu_uuid"):
        raise SystemExit(f"GPU {provenance['gpu_uuid']} is not the teacher table's {cost.get('gpu_uuid')}")
    policy = load_groot_libero_policy(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    runner = GrootStagedRunner(policy.model)
    if runner.live_schedule().schedule_id != PROFILE.denoise_schedule:
        raise SystemExit("encoder cost must be measured with the frozen teacher8 schedule")
    head = policy.model.action_head
    torch.manual_seed(args.seed)

    with runner.session():
        template = build_template(policy, runner, args.prompt)
        n_tokens = template.n_tokens
        if n_tokens != REPRESENTATIVE_N and not args.allow_other_shape:
            raise SystemExit(f"prompt gives N={n_tokens}, expected the representative {REPRESENTATIVE_N}")
        from openpi.cache.groot.staged import GrootStage1Output

        stage1 = GrootStage1Output(input_embeds=template.input_embeds, attention_mask=template.attention_mask,
                                   image_token_mask=template.image_token_mask, action_inputs=template.action_inputs)
        stage2 = runner.run_stage2_llm(stage1)
        features = stage2.backbone_features.clone()
        mask = stage2.attention_mask.clone()
        state = stage2.action_inputs["state"].clone()
        emb = stage2.action_inputs["embodiment_id"].clone()
        eager_src = runner.run_cp2_key_source(stage2)

        def encode(features_, mask_, state_, emb_):
            # Tensor-only twin of ``head.process_backbone_output`` (vlln ->
            # vl_self_attention; the mask is carried by the BatchFeature but
            # unused there). The production path is not compiled directly
            # because its ``BatchFeature`` (a UserDict) is a Dynamo graph
            # break whose resume frame is guarded on the object id of that
            # fresh dict: every call recompiled and the "CUDA-Graph" number
            # was ~370 ms of Dynamo, not the encoder (measured 2026-09-13).
            # ``fullgraph=True`` below makes any future break fail loudly.
            return head.vl_self_attention(head.vlln(features_)), head.state_encoder(state_, emb_)

        twin_vl, twin_state = encode(features, mask, state, emb)
        if not (torch.equal(twin_vl[0], eager_src.vl_encoded) and torch.equal(twin_state[0, -1], eager_src.state_encoded)):
            raise SystemExit("the compiled encoder twin is not bitwise the production run_cp2_key_source path")
        compiled = torch.compile(encode, mode=COMPILE_MODE, fullgraph=True)
        for _ in range(args.warmup):
            compiled(features, mask, state, emb)
        torch.cuda.synchronize()
        vl_c, st_c = compiled(features, mask, state, emb)
        torch.cuda.synchronize()
        parity = {
            "vl_max_abs_delta": float((vl_c[0].float() - eager_src.vl_encoded.float()).abs().max()),
            "state_max_abs_delta": float((st_c[0, -1].float() - eager_src.state_encoded.float()).abs().max()),
        }
        counters_before = inductor_counters()
        graphs = unique_graph_count(counters_before)
        skips_before = cudagraph_skips(counters_before)
        if graphs < 1:
            raise SystemExit("no compiled graph after warmup; refusing to label an eager number CUDA-Graph")
        # The record's identity (and therefore the NVTX marker the trace must
        # carry) is fixed *before* the measurement loop, as in the G-M cell.
        gpu = {**hardware_info(device), **provenance}
        sampling = {"n_tokens": n_tokens, "warmup": args.warmup, "iters": args.iters, "mode": COMPILE_MODE}
        record = {
            "protocol": libs.PROTOCOL, "record_kind": "cp2_encoder_cost", "teacher": PROFILE.name,
            "suite": args.suite, "schedule_id": PROFILE.denoise_schedule, "mode": COMPILE_MODE,
            "prompt": args.prompt, "prompt_sha256": _sha256_text(args.prompt), "n_tokens": n_tokens,
            "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
            "debug_sampling": sampling != libs.GROOT_E_SAMPLING,
            "graphs_per_call": graphs, "expected_cudagraph_launch_count": args.iters * graphs,
            "layout": {"kind": "groot_encoded_v1", "token_len": args.token_len,
                       "feature_dim": int(eager_src.vl_encoded.shape[1]),
                       "state_feat_dim": int(eager_src.state_encoded.shape[0])},
            "model": model_identity, "ckpt_weights_digest": model_identity["weights_digest"],
            "ckpt_sha256": ckpt_sha256, "denoising_steps": args.denoising_steps,
            "teacher_ckpt_sha256": cost.get("ckpt_sha256"),
            "teacher_cost_record": str(cost_path.resolve()),
            "teacher_cost_record_sha256": libs.sha256_file(cost_path),
            "hardware": gpu, "gpu_uuid": provenance["gpu_uuid"], "host": platform.node(),
            "torch": torch.__version__, "git_commit": libs.git_commit(),
            "ts": datetime.datetime.now().astimezone().isoformat(),
        }
        marker = encoder_trace_marker(record)
        samples: list[float] = []
        with cuda_profiler_range(), nvtx_measurement_range(marker):
            for _ in range(args.iters):
                _, ms = _timed(lambda: compiled(features, mask, state, emb))
                samples.append(ms)
        counters_after = inductor_counters()
    void: list[str] = [UNCERTIFIED]
    if unique_graph_count(counters_after) != graphs:
        void.append("recompilation during measurement")
    if cudagraph_skips(counters_after) != skips_before or skips_before:
        void.append("inductor cudagraph skip observed")
    record.update({
        "cp2_key_encoder_ms": float(statistics.median(samples)),
        "cp2_key_encoder_ms_p95": float(sorted(samples)[int(0.95 * (len(samples) - 1))]),
        "samples_ms": samples, "compiled_vs_eager": parity,
        "cudagraph_launch_count": None, "capture_calls_after_warmup": None,
        "inductor_counters_after": counters_after,
        "trace_marker": marker, "certified": False, "valid": False, "void_reasons": void,
    })
    _atomic_write_json(pathlib.Path(args.out), record)
    print(json.dumps({"cp2_key_encoder_ms": record["cp2_key_encoder_ms"], "n_tokens": n_tokens,
                      "graphs_per_call": graphs, "void_reasons": void}))
    return record


def certify_encoder(args: argparse.Namespace) -> dict:
    """Fill the CUDA-graph launch / capture counts from the nsys trace and certify the record.

    Only a record measured under the frozen sampling can be certified; a
    ``debug_sampling`` record (other shape, warmup or iteration count) is
    refused so a debugging run can never become the formal ``E``.
    """
    from exp.robocasa365.bench_groot_stages import parse_cuda_trace

    path = pathlib.Path(args.out)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("certified") is not False or UNCERTIFIED not in record.get("void_reasons", []):
        raise SystemExit(f"{path} is not a raw uncertified encoder record")
    sampling = {k: record.get(k) for k in libs.GROOT_E_SAMPLING}
    if record.get("debug_sampling") or sampling != libs.GROOT_E_SAMPLING:
        raise SystemExit(f"{path}: sampling {sampling} is not the frozen {libs.GROOT_E_SAMPLING}; "
                         "a debug measurement cannot be certified")
    marker = record.get("trace_marker")
    if marker != encoder_trace_marker(record):
        raise SystemExit("record trace marker does not match its identity")
    expected = record["expected_cudagraph_launch_count"]
    counts = parse_cuda_trace(pathlib.Path(args.cuda_trace), expected_marker=marker)
    record["cuda_trace_path"] = str(args.cuda_trace)
    record["cuda_trace_sha256"] = libs.sha256_file(args.cuda_trace)
    record["cudagraph_launch_count"] = counts["cudagraph_launch_count"]
    record["capture_calls_after_warmup"] = counts["graph_capture_calls"]
    reasons = [r for r in record.get("void_reasons", []) if r != UNCERTIFIED]
    if counts["cudagraph_launch_count"] <= 0:
        reasons.append("trace shows no cudaGraphLaunch -- graphs were never replayed")
    if counts["cudagraph_launch_count"] != expected:
        reasons.append(f"cudaGraphLaunch count mismatch: expected {expected}, got {counts['cudagraph_launch_count']}")
    if counts["graph_capture_calls"]:
        reasons.append(f"measurement trace contains graph capture APIs: count={counts['graph_capture_calls']}")
    record["void_reasons"] = reasons
    record["valid"] = not reasons
    record["certified"] = not reasons
    _atomic_write_json(path, record)
    print(json.dumps({"certified": record["certified"], "void_reasons": reasons,
                      "cp2_key_encoder_ms": record["cp2_key_encoder_ms"]}))
    if reasons:
        raise SystemExit(1)
    return record


# ------------------------------------------------------------------
# decision overhead
# ------------------------------------------------------------------


def internal_cp2_yaml(library_pkl: str, proj_meta: dict, out_dir: pathlib.Path) -> pathlib.Path:
    """An n0 arm at the paper's reference threshold, for the bench only (not an experiment arm)."""
    projection = libs.ProjectionArgs.from_projection_meta(proj_meta)
    doc = cp2_arm_yaml(preload_path=str(pathlib.Path(library_pkl).resolve()), projection=projection,
                       tier="n0", theta_raw=PROFILE.reference_theta_raw, profile=PROFILE)
    path = out_dir / "bench_internal_cp2_n0.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def run_overhead(args: argparse.Namespace) -> dict:
    """The real per-decision CP2 overhead over the accepted cohort (plan §3.11 preflight).

    Boundary per decision: synchronize -> ``run_cp2_key_source`` (session) ->
    ``check(CP2)`` -> synchronize, with stage 1 reconstruction and stage 2
    outside it. Writes ``per_decision.csv`` and ``overhead.json`` (cold = the
    first ``--cold`` decisions, warm = the rest, per-segment probe statistics,
    the verdict from the warm total P95); fails on a missing probe or on
    ``<= 50`` decisions.
    """
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = json.loads(pathlib.Path(args.accepted_manifest).read_text(encoding="utf-8"))
    if acc.get("suite") != args.suite or not acc.get("ok") or not acc.get("task_map_bound"):
        raise SystemExit("--accepted-manifest is not an accepted, task-map-bound cohort for this suite")
    lib_meta = {k: v for k, v in libs.load_pickle(args.library_pkl).items() if k not in ("entries", "library_stats", "prompt_pool")}
    if lib_meta.get("key_builder_type") != KEY_BUILDER_TYPE or lib_meta.get("stage1_path") != STAGE1_PATH:
        raise SystemExit("--library-pkl is not a GR00T CP2 library built through the reconstructed template")
    cache_yaml = internal_cp2_yaml(args.library_pkl, lib_meta["projection"], out_dir)
    config, comps, orch, timer = build_orchestrator(cache_yaml)
    model_binding = libs.assert_model_binding(comps["storage"].artifact_meta.get("model"), args.checkpoint)
    policy = load_groot_libero_policy(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    runner = GrootStagedRunner(policy.model, timer=timer)
    if runner.live_schedule().schedule_id != lib_meta.get("schedule_id"):
        raise SystemExit("served head schedule != library schedule")
    templates = TemplateCache(policy, runner)

    episodes = list(acc["accepted"])
    rows: list[list] = []
    totals: list[float] = []
    seg_samples: dict[str, list[float]] = {s: [] for s in ("cp2_encode", *SEGMENTS)}
    checks: list[float] = []
    n = 0
    orch.on_task_begin()
    current = None
    for ep in episodes:
        if n >= args.max_decisions:
            break
        h5_path = pathlib.Path(ep["h5"])
        if libs.sha256_file(h5_path) != ep["h5_sha256"]:
            raise SystemExit(f"{h5_path}: sha256 changed since acceptance")
        with h5py.File(h5_path, "r") as f:
            task = h5_task(f)
            if task != ep["task_language"]:
                raise SystemExit(f"{h5_path}: instruction {task!r} != accepted {ep['task_language']!r}")
            with runner.session():
                template = templates.get(task)
            steps = list(libs.iter_steps(f))
            for step_idx, group in steps:
                if n >= args.max_decisions:
                    break
                if current != h5_path.stem:
                    if current is not None:
                        orch.on_episode_end()
                    orch.on_episode_start(task_key=task, episode_id=h5_path.stem)
                    current = h5_path.stem
                # Outside the timed boundary: rebuild stage 1 and run stage 2.
                with runner.session():
                    stage1 = reconstruct_stage1(template, group)
                    stage2 = runner.run_stage2_llm(stage1)
                timer.on_task_begin()
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                t0 = time.perf_counter()
                with runner.session():
                    source = runner.run_cp2_key_source(stage2)
                t_mid = time.perf_counter()
                orch.check(CheckpointID.CP2, stage2=stage2, cp2_source=source)
                torch.cuda.synchronize(device) if device.type == "cuda" else None
                t1 = time.perf_counter()
                orch.clear()
                source = None
                total_ms = (t1 - t0) * 1000.0
                check_ms = (t1 - t_mid) * 1000.0
                per = timer.summary(task_only=True)
                row = [h5_path.stem, step_idx, total_ms, check_ms]
                for s in ("cp2_encode", *SEGMENTS):
                    st = per.get(s)
                    val = float(st.mean_ms) if st is not None else None
                    if val is not None:
                        seg_samples[s].append(val)
                    row.append(val)
                missing = [s for s in CORE_SEGMENTS_GROOT if per.get(s) is None]
                if missing:
                    raise SystemExit(f"decision {h5_path.stem}:{step_idx} recorded no {missing} probe — timer not wired")
                totals.append(total_ms)
                checks.append(check_ms)
                rows.append(row)
                n += 1
    if current is not None:
        orch.on_episode_end()
    orch.on_task_end()
    if n <= 50:
        raise SystemExit(f"only {n} decisions; the preflight needs more than 50 real-model samples")
    with (out_dir / "per_decision.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "step_idx", "total_ms", "check_total_ms", *[f"{s}_ms" for s in ("cp2_encode", *SEGMENTS)]])
        w.writerows(rows)
    cold, warm = totals[: args.cold], totals[args.cold:]
    meta = comps["storage"].artifact_meta or {}
    record = {
        "protocol": libs.PROTOCOL, "record_kind": "cp2_decision_overhead", "teacher": PROFILE.name,
        "suite": args.suite, "cache_yaml": str(cache_yaml.resolve()),
        "config_digest": hashlib.sha256(cache_yaml.read_bytes()).hexdigest(),
        "library": config.backend.in_memory.preload_path, "library_sha256": meta.get("library_sha256"),
        "library_entries": meta.get("entry_count"), "projection": meta.get("projection"),
        "stage1_path": lib_meta.get("stage1_path"), "schedule_id": meta.get("schedule_id"),
        "model": {"library_model": meta.get("model"), "bound": model_binding},
        "accepted_manifest": str(pathlib.Path(args.accepted_manifest).resolve()),
        "accepted_manifest_sha256": libs.sha256_file(args.accepted_manifest),
        "hardware": hardware_info(device), "n_decisions": n, "cold_decisions": len(cold),
        "cold": _pctl(cold), "warm": _pctl(warm),
        "check_total": _pctl(checks[args.cold:]),
        "per_segment": {s: _pctl(v) for s, v in seg_samples.items()},
        "thresholds_ms": {"ok": 10.0, "halt": 40.0},
        "verdict": verdict_for(_pctl(warm)["p95"]),
        "timer_enabled": bool(getattr(timer, "_enabled", False)),
        "boundary": "synchronize -> run_cp2_key_source (session) -> check(CP2) -> synchronize; stage 2 outside",
        "git_commit": libs.git_commit(), "ts": datetime.datetime.now().astimezone().isoformat(),
    }
    libs.dump_json(out_dir / "overhead.json", record)
    print(json.dumps({k: record[k] for k in ("n_decisions", "cold", "warm", "check_total", "verdict")}))
    return record


def main() -> None:
    """CLI entry: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=("encoder-cost", "certify-encoder", "overhead"))
    ap.add_argument("--suite", default="", choices=["", *sorted(libs.SUITE_TAGS)])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--cost-record", default="", help="encoder-cost: the certified teacher cost table")
    ap.add_argument("--out", default="", help="encoder-cost / certify-encoder: the encoder cost JSON")
    ap.add_argument("--cuda-trace", default="", help="certify-encoder: nsys CSV export")
    ap.add_argument("--prompt", default=REPRESENTATIVE_PROMPT)
    ap.add_argument("--allow-other-shape", action="store_true")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-len", type=int, default=640)
    ap.add_argument("--library-pkl", default="", help="overhead: the CP2 library")
    ap.add_argument("--accepted-manifest", default="", help="overhead: accepted_shadow_manifest.json")
    ap.add_argument("--out-dir", default="", help="overhead: output directory")
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--cold", type=int, default=50)
    args = ap.parse_args()
    if args.mode == "encoder-cost":
        for f in ("suite", "checkpoint", "cost_record", "out"):
            if not getattr(args, f):
                ap.error(f"--{f.replace('_', '-')} is required for encoder-cost")
        measure_encoder_cost(args)
    elif args.mode == "certify-encoder":
        if not (args.out and args.cuda_trace):
            ap.error("--out and --cuda-trace are required for certify-encoder")
        certify_encoder(args)
    else:
        for f in ("suite", "checkpoint", "library_pkl", "accepted_manifest", "out_dir"):
            if not getattr(args, f):
                ap.error(f"--{f.replace('_', '-')} is required for overhead")
        run_overhead(args)


if __name__ == "__main__":
    main()
