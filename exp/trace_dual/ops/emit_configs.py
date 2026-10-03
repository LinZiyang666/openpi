"""Emit the 8 arm yamls + single-arm matrices of the trace dual-model run.

Retrieval sections are copied verbatim from the canonical yamls; only the
cp1 gate/judge are replaced: always_search + threshold judge at +1e6 (never
hits -> served = full inference, cache in shadow) or -1e6 (always FULL_HIT ->
served = top-1 cached action, inference in shadow). No warm tiers.
"""
import copy, pathlib, yaml, hashlib, json

REPO = pathlib.Path("/home/weiland/projects/openpi")
OUT = pathlib.Path("/home/weiland/trace_runs/dual_20260923/config")
SRC = {
    ("pi05", "libero_spatial"): REPO / "exp/gate_threshold_pareto/config/libero_spatial/eval/gtp_ws_sp_fh30.yaml",
    ("pi05", "libero_10"): REPO / "exp/gate_threshold_pareto/config/libero_10/eval/gtp_ws_l10_fh30.yaml",
    ("groot", "libero_spatial"): REPO / "exp/libero_groot/config/rit/libero_spatial/source_template.yaml",
    ("groot", "libero_10"): REPO / "exp/libero_groot/config/rit/libero_10/source_template.yaml",
}
TAG = {"libero_spatial": "sp", "libero_10": "l10"}
MODES = {"inf": 1.0e6, "cache": -1.0e6}

rows = []
for (model, suite), src in SRC.items():
    base = yaml.safe_load(src.read_text())
    pp = base["backend"]["in_memory"]["preload_path"]
    if model == "pi05":
        # Authoritative cp1_spatial_pool_16 (sha 36cd0f3b / f13517ad), copied into the run
        # root; the main-tree copies are the stale WSL ones (same entries, different bytes).
        base["backend"]["in_memory"]["preload_path"] = f"/home/weiland/trace_runs/dual_20260923/libs/{suite}/cp1_spatial_pool_16.pkl"
    elif not pp.startswith("/"):
        base["backend"]["in_memory"]["preload_path"] = str(REPO / pp)
    for mode, thr in MODES.items():
        cfg = copy.deepcopy(base)
        cp1 = cfg["checkpoints"]["cp1"]
        cp1["gate"] = {"type": "always_search"}
        cp1["judge"] = {"type": "threshold", "threshold": thr}
        cfg["write_policy"] = {"type": "never"}
        if model == "pi05":
            # debug fully on: the only record_* switch that defaults off (pi0.5-only;
            # the GR00T trace path has no model-image capture). out_dir / build-cache
            # come from the server CLI.
            cfg["trace"] = {"record_model_images": True}
        arm = f"tr_{model}_{TAG[suite]}_{mode}"
        p = OUT / f"{arm}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        m = OUT / f"matrix_{arm}.yaml"
        m.write_text(yaml.safe_dump({"arms": [{"arm": arm, "yaml": f"run_cfg/{arm}.yaml", "suite": suite}]}, sort_keys=False))
        rows.append({"arm": arm, "model": model, "suite": suite, "mode": mode, "src": str(src),
                     "src_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                     "yaml_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT / "arms.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=1))
