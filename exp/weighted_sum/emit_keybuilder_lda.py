"""Emit the key-builder x LDA supplement: fit templates first, then the served arms.

The supplement puts every dimensionality reduction that the original pure-cache
exploration compared (``cp1_mean_pool`` / ``cp1_max_pool`` / ``cp1_spatial_pool_64``
/ CLIP ViT-B/32) under the same closed-form fusion rule (phase-discriminant LDA,
``lcw_fit_weights.py``) and adds one single-key arm that does not split the
prefix into modalities at all (``cp1_llm_layer_extract`` + ``prefix_mean_pool``,
layer 0; it has one scored field, so there is no weight to choose). Every arm
is the historical pure-cache recipe: ``always_search`` + ``always_hit`` +
``top_k 1`` + the library's own LOEO normalizers + ``write_policy: never``.

Two stages, run in this order, with the LDA fits in between so that neither
stage depends on the other's output being generated in the same process:

``templates``
    For each suite and pool/CLIP library: a three-field template (v0/v1/rs all
    enabled at 1/3, normalizers = the library's ``selected`` Phase-1 entry,
    which must be zscore+tanh with finite mu and positive sigma) that
    ``lcw_fit_weights.py`` reads for the normalizer parameters, plus the
    diagnostic cells file it needs (the spatial_16 A-pool grid converted from
    ``summary_grid6.json``; diagnostic output only, never part of the LDA
    solution). It prints the fit command per library.

``final``
    Reads the finished fits (``lda@0.05``), rejects degenerate ones, writes one
    yaml per arm, the LLM arm, the two matrices per suite, ``weights.json``
    (library / calibration / template identities and the LDA intermediates)
    and ``active_manifest.json`` (the strict input contract for
    ``lcw_ablation_summary.py --input-manifest``). Each yaml is loaded back
    through ``load_cache_config`` and asserted field by field.

Usage (from the repo root):
  uv run python exp/weighted_sum/emit_keybuilder_lda.py templates
  # run the printed lcw_fit_weights.py commands
  uv run python exp/weighted_sum/emit_keybuilder_lda.py final [--skip-clip --skip-clip-reason ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pathlib
import pickle

import sys

import yaml

from openpi.cache.config import load_cache_config

# Runnable by path from anywhere: ``exp`` is a package rooted three levels up.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from exp.weighted_sum.emit_yamls import build_eval_config  # noqa: E402

SUITES = ("libero_spatial", "libero_10")
POOL_STEMS = ("cp1_mean_pool", "cp1_max_pool", "cp1_spatial_pool_64", "clip_vit_b_32")
CLIP_STEM = "clip_vit_b_32"
LLM_STEM = "cp1_llm_l0_prefix_mean_pool"
LLM_SUBDIR = "llm_layer_extract"
FIELDS = ("vision_0", "vision_1", "robot_state")
LDA_FIT = "lda@0.05"
LDA_DETAIL = "detail@0.05"
GRID6_STEP = 6

# builder_type the calibration entry must carry per stem; the artifact stem is
# the key everywhere (two CLIP variants share builder_type "clip").
BUILDER_OF = {
    "cp1_mean_pool": "cp1_mean_pool",
    "cp1_max_pool": "cp1_max_pool",
    "cp1_spatial_pool_64": "cp1_spatial_pool_64",
    CLIP_STEM: "clip",
    LLM_STEM: "cp1_llm_layer_extract",
}
CLIP_VARIANT = {"clip_model_name": "ViT-B-32", "clip_pretrained": "openai"}
LLM_VECTOR_DIMS = {"vision_0": 2048, "robot_state": 32}
LLM_KEY_BUILDER = {"type": "cp1_llm_layer_extract", "extract_layer": 0,
                   "prefix_reducer": {"type": "prefix_mean_pool"}}

LOCAL_ART = pathlib.Path("exp/common/data/cache_artifacts")
SERVED_ART = "/home/weiland/openpi/exp/common/data/cache_artifacts"
BASE_CALIB = "exp/weighted_sum/data/{suite}/phase1/calibration_normalizers.json"
DATA = pathlib.Path("exp/weighted_sum/data/keybuilder_lda")
CFG = pathlib.Path("exp/weighted_sum/config/keybuilder_lda")
GRID6_SUMMARY = pathlib.Path("exp/weighted_sum/data/grid6/summary_grid6.json")
FIT_SCRIPT = "exp/weighted_sum/analysis/lcw_fit_weights.py"
REFERENCE_DIR = "exp/weighted_sum/data/fusion_ablation/pi05/{suite}"
REFERENCE_ARM = "fw_pi05_{suite}_lda"
APOOL_RECORD = "exp/ablation_study/cache_size/config/apool_{suite}.yaml"


# ------------------------------------------------------------------
# Paths and identities
# ------------------------------------------------------------------
def library_rel(suite: str, stem: str) -> str:
    """Library path relative to the artifact root (same layout locally and on the server)."""
    sub = f"{LLM_SUBDIR}/" if stem == LLM_STEM else ""
    return f"{suite}/{sub}{stem}.pkl"


def local_library(suite: str, stem: str, root: pathlib.Path = LOCAL_ART) -> pathlib.Path:
    return root / library_rel(suite, stem)


def served_library(suite: str, stem: str, root: str = SERVED_ART) -> str:
    return f"{root}/{library_rel(suite, stem)}"


def yaml_id(suite: str, stem: str) -> str:
    """Globally unique arm name: the suite is part of it."""
    return f"kb_{suite}_{stem}" + ("" if stem == LLM_STEM else "_lda")


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------
def resolve_calibration(suite: str, stem: str, *, supplement_dir: pathlib.Path,
                        base: str | None = None) -> tuple[dict, pathlib.Path]:
    """The Phase-1 entry for one library: the supplement file if one was calibrated, else the base file.

    Base calibrations are the phase-1 JSONs the grid was served with; the
    supplement directory (``<data>/calibration``) holds the libraries
    calibrated for this experiment only (libero_10 CLIP, both LLM libraries).
    A supplement file wins when it exists so a re-calibration is never
    shadowed silently by the base.
    """
    base = BASE_CALIB if base is None else base
    supp = supplement_dir / suite / f"{stem}.json"
    if supp.exists():
        entry = json.loads(supp.read_text())[stem]
        path = supp
    else:
        path = pathlib.Path(base.format(suite=suite))
        calib = json.loads(path.read_text())
        if stem not in calib:
            raise SystemExit(f"{suite}/{stem}: no calibration in {path} and no supplement at {supp}")
        entry = calib[stem]
    if entry["builder_type"] != BUILDER_OF[stem]:
        raise SystemExit(f"{suite}/{stem}: calibration builder_type {entry['builder_type']!r} "
                         f"!= {BUILDER_OF[stem]!r}")
    return entry, path


def check_zscore_tanh(fields_calib: dict, fields=FIELDS, *, who: str) -> None:
    """Compatibility gate for the three-field LDA: every scored field is zscore+tanh with usable mu/sigma."""
    for f in fields:
        if f not in fields_calib:
            raise SystemExit(f"{who}: field {f!r} has no calibration")
        sel = fields_calib[f]["selected"]
        p = sel["params"]
        if sel["method"] != "zscore" or p.get("squash", "tanh") != "tanh":
            raise SystemExit(f"{who}: field {f!r} selected {sel['method']}/{p.get('squash')}, "
                             "the LDA fit assumes zscore+tanh")
        if not finite([p["mu"], p["sigma"]]) or p["sigma"] <= 0:
            raise SystemExit(f"{who}: field {f!r} has unusable mu/sigma {p['mu']}/{p['sigma']}")


# ------------------------------------------------------------------
# Stage 1: templates
# ------------------------------------------------------------------
def diagnostic_cells(summary: dict, suite: str, *, source: str) -> dict:
    """``summary_grid6.json`` cells (``"a/b/c": sr``) -> the cells shape ``lcw_fit_weights.py`` reads.

    These are the spatial_16 A-pool grid readings; the fit script only uses
    them to print nearest-cell diagnostics, so their leader/SR must never be
    reported as a result of the builder being fitted.
    """
    cells = summary[f"pi05_{suite}"]["cells"]
    rows = []
    for key, sr in cells.items():
        a, b, c = (int(x) for x in key.split("/"))
        if a + b + c != GRID6_STEP:
            raise SystemExit(f"{suite}: cell {key} is not on the 1/{GRID6_STEP} simplex")
        rows.append({"w": [a / GRID6_STEP, b / GRID6_STEP, c / GRID6_STEP], "sr": float(sr), "n": 500})
    return {"cells": rows, "source": source, "library": "cp1_spatial_pool_16",
            "note": "diagnostic only: nearest-cell readings of another builder's grid, not this builder's result"}


def template_config(suite: str, stem: str, *, served_root: str, data: pathlib.Path) -> tuple[dict, dict, pathlib.Path]:
    entry, calib_path = resolve_calibration(suite, stem, supplement_dir=data / "calibration")
    check_zscore_tanh(entry["fields"], who=f"{suite}/{stem}")
    cfg = build_eval_config(
        builder_type=entry["builder_type"],
        vector_dims=entry["vector_dims"],
        preload_path=served_library(suite, stem, served_root),
        weights={f: 1 / 3 for f in FIELDS},
        fields_calib=entry["fields"],
    )
    sn = cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]
    if set(sn) != set(FIELDS):
        raise SystemExit(f"{suite}/{stem}: template normalizers {sorted(sn)} != {sorted(FIELDS)}")
    return cfg, entry, calib_path


def run_templates(args) -> None:
    summary = json.loads(args.grid6_summary.read_text())
    cells_dir = args.data / "diagnostic_cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for suite in SUITES:
        cells_path = cells_dir / f"{suite}.json"
        cells_path.write_text(json.dumps(diagnostic_cells(summary, suite, source=str(args.grid6_summary)),
                                         indent=1) + "\n")
        tdir = args.cfg / suite / "templates"
        tdir.mkdir(parents=True, exist_ok=True)
        for stem in POOL_STEMS:
            cfg, _entry, calib_path = template_config(suite, stem, served_root=args.served_root, data=args.data)
            tpath = tdir / f"{stem}_template.yaml"
            tpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
            lib = local_library(suite, stem, args.local_root)
            commands.append(
                f"uv run python {FIT_SCRIPT} --library {lib} --template {tpath} "
                f"--cells {cells_path} --output {args.data / 'fits' / suite / stem}")
            print(f"{suite:15s} {stem:22s} template={tpath} calibration={calib_path}")
    print("\n# LDA fits, one per library:")
    for c in commands:
        print(c)


# ------------------------------------------------------------------
# Stage 2: final arms
# ------------------------------------------------------------------
def read_fit(fit_path: pathlib.Path, *, library: pathlib.Path, template: pathlib.Path) -> dict:
    """Load one ``lcw_fit_weights.json`` and reject anything that is not a usable LDA solution of these inputs.

    The fit must name the library and the template by content (sha256): a
    library rebuilt at the same path or a re-calibrated template invalidates
    the fit, and a fit without that identity is not accepted at all.
    """
    if not fit_path.exists():
        raise SystemExit(f"missing fit {fit_path}; run the templates stage commands first")
    fit = json.loads(fit_path.read_text())
    if pathlib.Path(fit.get("library", "")).resolve() != library.resolve():
        raise SystemExit(f"{fit_path}: fitted on {fit.get('library')}, expected {library}")
    if not fit.get("library_sha256"):
        raise SystemExit(f"{fit_path}: no library_sha256; refit with the current lcw_fit_weights.py")
    if fit["library_sha256"] != sha256_of(library):
        raise SystemExit(f"{fit_path}: library {library} changed since the fit (sha mismatch)")
    tpl_rec = fit.get("template")
    if not tpl_rec or "sha256" not in tpl_rec or "path" not in tpl_rec:
        raise SystemExit(f"{fit_path}: no template identity; refit with the current lcw_fit_weights.py")
    if pathlib.Path(tpl_rec["path"]).resolve() != template.resolve():
        raise SystemExit(f"{fit_path}: fitted with template {tpl_rec['path']}, expected {template}")
    if tpl_rec["sha256"] != sha256_of(template):
        raise SystemExit(f"{fit_path}: template {template} changed since the fit (sha mismatch)")
    lda = fit["fits"].get(LDA_FIT)
    detail = fit["fits"].get(LDA_DETAIL, {}).get("fisher_lda")
    if lda is None or detail is None or lda.get("w") is None or lda.get("raw") is None:
        raise SystemExit(f"{fit_path}: no usable {LDA_FIT} / {LDA_DETAIL} entry")
    w, raw = lda["w"], lda["raw"]
    if len(w) != len(FIELDS) or not finite(w) or min(w) < 0 or abs(sum(w) - 1.0) > 1e-6:
        raise SystemExit(f"{fit_path}: {LDA_FIT} weights {w} are not a point on the simplex")
    if max(w) <= 0:
        raise SystemExit(f"{fit_path}: {LDA_FIT} weights are all zero")
    if len(raw) != len(FIELDS) or not finite(raw) or not finite(detail["d"]) or not finite(detail["lda_raw"]) \
            or not all(finite(row) for row in detail["cov"]):
        raise SystemExit(f"{fit_path}: non-finite LDA intermediates")
    if list(detail["lda_raw"]) != list(raw):
        raise SystemExit(f"{fit_path}: {LDA_FIT}.raw differs from {LDA_DETAIL}.fisher_lda.lda_raw")
    clipped = [max(x, 0.0) for x in raw]
    if sum(clipped) <= 0:
        raise SystemExit(f"{fit_path}: raw LDA direction {raw} has no non-negative mass; the fit is degenerate")
    expect_w = [x / sum(clipped) for x in clipped]
    if any(abs(a - b) > 1e-9 for a, b in zip(w, expect_w)):
        raise SystemExit(f"{fit_path}: w {w} is not the clipped, normalised raw direction {expect_w}")
    if int(detail.get("queries", 0)) <= 0:
        raise SystemExit(f"{fit_path}: no query contributed both classes; the fit is empty")
    return fit


def normalizers_of(cfg: dict) -> dict:
    return cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]


def pool_arm(suite: str, stem: str, w: list[float], *, served_root: str, template: pathlib.Path,
             data: pathlib.Path) -> tuple[dict, dict, pathlib.Path]:
    entry, calib_path = resolve_calibration(suite, stem, supplement_dir=data / "calibration")
    check_zscore_tanh(entry["fields"], who=f"{suite}/{stem}")
    cfg = build_eval_config(
        builder_type=entry["builder_type"],
        vector_dims=entry["vector_dims"],
        preload_path=served_library(suite, stem, served_root),
        weights=dict(zip(FIELDS, w)),
        fields_calib=entry["fields"],
    )
    # The scored fields must carry exactly the normalizers the fit saw. A
    # zero-weight field drops its normalizer (existing emitter behaviour); the
    # template itself still had all three.
    tpl_norm = normalizers_of(yaml.safe_load(template.read_text()))
    for f, nf in normalizers_of(cfg).items():
        if nf != tpl_norm[f]:
            raise SystemExit(f"{suite}/{stem}: normalizer for {f} differs from the fit template")
    return cfg, entry, calib_path


def llm_arm(suite: str, *, served_root: str, data: pathlib.Path) -> tuple[dict, dict, pathlib.Path]:
    """Single-key arm: the whole prefix after LLM layer 0, mean-pooled into ``vision_0``; state at weight 0."""
    entry, calib_path = resolve_calibration(suite, LLM_STEM, supplement_dir=data / "calibration")
    if entry["vector_dims"] != LLM_VECTOR_DIMS:
        raise SystemExit(f"{suite}/{LLM_STEM}: vector_dims {entry['vector_dims']} != {LLM_VECTOR_DIMS}")
    cfg = build_eval_config(
        builder_type=BUILDER_OF[LLM_STEM],
        vector_dims=entry["vector_dims"],
        preload_path=served_library(suite, LLM_STEM, served_root),
        weights={"vision_0": 1.0},
        fields_calib=entry["fields"],
    )
    cfg["key_builder"] = dict(LLM_KEY_BUILDER)
    return cfg, entry, calib_path


class _UnknownKeyCatcher(logging.Handler):
    """``_dict_to_dataclass`` only warns on unknown yaml keys; here that is a failure."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.unknown: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Unknown config key" in msg:
            self.unknown.append(msg)


def accept_yaml(path: pathlib.Path, *, builder_type: str, weights: dict[str, float],
                normalizers: dict, vector_dims: dict, preload_path: str,
                key_builder_extra: dict | None = None) -> None:
    """Load the yaml through the production loader and assert every value this experiment relies on."""
    catcher = _UnknownKeyCatcher()
    log = logging.getLogger("openpi.cache.config")
    log.addHandler(catcher)
    try:
        cfg = load_cache_config(path)
    finally:
        log.removeHandler(catcher)
    if catcher.unknown:
        raise SystemExit(f"{path}: unknown keys: {catcher.unknown}")
    if cfg.key_builder.type != builder_type:
        raise SystemExit(f"{path}: key_builder.type {cfg.key_builder.type!r} != {builder_type!r}")
    for k, v in (key_builder_extra or {}).items():
        actual = getattr(cfg.key_builder, k)
        actual = actual.type if k == "prefix_reducer" else actual
        expect = v["type"] if k == "prefix_reducer" else v
        if actual != expect:
            raise SystemExit(f"{path}: key_builder.{k} {actual!r} != {expect!r}")
    if cfg.routing is not None:
        raise SystemExit(f"{path}: routing section present")
    cps = [c for c in cfg.checkpoints if not c.startswith("_")]
    cp1 = cfg.checkpoints["cp1"]
    if [c for c in cps if cfg.checkpoints[c].enabled] != ["cp1"]:
        raise SystemExit(f"{path}: enabled checkpoints {cps} != ['cp1']")
    if cp1.gate.type != "always_search" or cp1.judge.type != "always_hit":
        raise SystemExit(f"{path}: gate/judge {cp1.gate.type}/{cp1.judge.type} is not the pure-cache recipe")
    ss = cp1.search_strategy
    if ss.type != "weighted_score_sum_knn" or ss.top_k != 1 or ss.trajectory_depth != 1 or not ss.task_scoped:
        raise SystemExit(f"{path}: search_strategy {ss.type}/top_k={ss.top_k}/depth={ss.trajectory_depth}"
                         f"/task_scoped={ss.task_scoped} is not the single-step task-scoped recipe")
    if cfg.write_policy.type != "never":
        raise SystemExit(f"{path}: write_policy {cfg.write_policy.type} != never")
    if cfg.backend.type != "in_memory" or cfg.backend.in_memory.preload_path != preload_path:
        raise SystemExit(f"{path}: backend {cfg.backend.type} preload {cfg.backend.in_memory.preload_path}")
    if dict(cfg.backend.vector_dims) != {k: int(v) for k, v in vector_dims.items()}:
        raise SystemExit(f"{path}: vector_dims {dict(cfg.backend.vector_dims)} != {vector_dims}")
    served = {}
    for name in ("vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state", "vlm_out"):
        kf = getattr(cfg.keys, name)
        if kf.enabled:
            served[name] = float(kf.weight)
    # A zero-weight field is dropped by the emitter unless the builder requires
    # it, in which case it stays enabled at weight 0 (no vote, no normalizer).
    expect = {f: float(w) for f, w in weights.items() if w > 0}
    for req in ("vision_0", "robot_state"):
        expect.setdefault(req, 0.0)
    if served != expect:
        raise SystemExit(f"{path}: served weights {served} != {expect}")
    if abs(sum(served.values()) - 1.0) > 1e-6:
        raise SystemExit(f"{path}: served weights sum to {sum(served.values())}")
    sn = ss.score_normalization
    if sn is None or sn.type != "per_field" or dict(sn.fields) != normalizers:
        raise SystemExit(f"{path}: score_normalization differs from the recorded normalizers")
    scored = {f for f, w in served.items() if w > 0}
    if set(sn.fields) != scored:
        raise SystemExit(f"{path}: normalizer fields {sorted(sn.fields)} != scored fields {sorted(scored)}")


def library_identity(path: pathlib.Path, *, stem: str, expect_dims: dict) -> dict:
    """Metadata read from the local library: dims and (for CLIP) the model variant the online builder must match."""
    if not path.exists():
        raise SystemExit(f"library {path} is not on this host")
    with path.open("rb") as f:
        art = pickle.load(f)
    dims = {k: int(v) for k, v in art["vector_dims"].items()}
    if dims != {k: int(v) for k, v in expect_dims.items()}:
        raise SystemExit(f"{path}: vector_dims {dims} != calibration {expect_dims}")
    if art.get("key_builder_type") != BUILDER_OF[stem]:
        raise SystemExit(f"{path}: key_builder_type {art.get('key_builder_type')!r} != {BUILDER_OF[stem]!r}")
    ident = {"path": str(path), "sha256": sha256_of(path), "entries": len(art["entries"]),
             "trajectories": len({e.trajectory_id for e in art["entries"]}), "vector_dims": dims}
    if stem == CLIP_STEM:
        variant = {k: art.get(k) for k in CLIP_VARIANT}
        if variant != CLIP_VARIANT:
            raise SystemExit(f"{path}: CLIP variant {variant} != online default {CLIP_VARIANT}")
        ident["clip"] = variant
    return ident


def run_final(args) -> None:
    pool_stems = [s for s in POOL_STEMS if not (args.skip_clip and s == CLIP_STEM)]
    weights_out: dict = {}
    manifest: dict = {"suites": {}, "reference_arm_pattern": REFERENCE_ARM,
                      "clip_deferred": bool(args.skip_clip),
                      "clip_deferred_reason": args.skip_clip_reason if args.skip_clip else None}
    all_ids: list[str] = []
    for suite in SUITES:
        out_dir = args.cfg / suite
        out_dir.mkdir(parents=True, exist_ok=True)
        rows_pool, rows_llm = [], []
        yaml_records: dict = {}
        libraries: dict = {}

        for stem in pool_stems:
            template = out_dir / "templates" / f"{stem}_template.yaml"
            lib = local_library(suite, stem, args.local_root)
            fit_path = args.data / "fits" / suite / stem / "lcw_fit_weights.json"
            fit = read_fit(fit_path, library=lib, template=template)
            w = [float(x) for x in fit["fits"][LDA_FIT]["w"]]
            cfg, entry, calib_path = pool_arm(suite, stem, w, served_root=args.served_root, template=template,
                                              data=args.data)
            arm = yaml_id(suite, stem)
            path = out_dir / f"{arm}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            accept_yaml(path, builder_type=entry["builder_type"], weights=dict(zip(FIELDS, w)),
                        normalizers=normalizers_of(cfg), vector_dims=entry["vector_dims"],
                        preload_path=served_library(suite, stem, args.served_root))
            libraries[stem] = library_identity(lib, stem=stem, expect_dims=entry["vector_dims"])
            if (libraries[stem]["sha256"] != fit["library_sha256"]
                    or sha256_of(template) != fit["template"]["sha256"]):
                raise SystemExit(f"{suite}/{stem}: library or template changed while emitting; regenerate the arm")
            detail = fit["fits"][LDA_DETAIL]["fisher_lda"]
            weights_out[arm] = {
                "suite": suite, "stem": stem, "builder_type": entry["builder_type"],
                "library": libraries[stem]["path"], "library_sha256": fit["library_sha256"],
                "calibration": str(calib_path), "calibration_sha256": sha256_of(calib_path),
                "selected": {f: entry["fields"][f]["selected"] for f in FIELDS},
                "template": str(template), "template_sha256": fit["template"]["sha256"],
                "fit": str(fit_path), "fit_sha256": sha256_of(fit_path),
                "lda": {"fields": list(FIELDS), "d": detail["d"], "cov": detail["cov"],
                        "lda_raw": detail["lda_raw"], "queries": detail["queries"], "w": w},
            }
            rows_pool.append({"arm": arm, "yaml": str(path), "sidecar": None})
            yaml_records[arm] = {"path": str(path), "sha256": sha256_of(path), "library": stem, "w": w}
            print(f"{arm:44s} w=" + "/".join(f"{x:.4f}" for x in w))

        cfg, entry, calib_path = llm_arm(suite, served_root=args.served_root, data=args.data)
        arm = yaml_id(suite, LLM_STEM)
        path = out_dir / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        accept_yaml(path, builder_type=BUILDER_OF[LLM_STEM], weights={"vision_0": 1.0},
                    normalizers=normalizers_of(cfg), vector_dims=entry["vector_dims"],
                    preload_path=served_library(suite, LLM_STEM, args.served_root),
                    key_builder_extra={"extract_layer": 0, "prefix_reducer": {"type": "prefix_mean_pool"}})
        lib = local_library(suite, LLM_STEM, args.local_root)
        libraries[LLM_STEM] = library_identity(lib, stem=LLM_STEM, expect_dims=entry["vector_dims"])
        weights_out[arm] = {
            "suite": suite, "stem": LLM_STEM, "builder_type": BUILDER_OF[LLM_STEM],
            "library": libraries[LLM_STEM]["path"], "library_sha256": libraries[LLM_STEM]["sha256"],
            "calibration": str(calib_path), "calibration_sha256": sha256_of(calib_path),
            "selected": {"vision_0": entry["fields"]["vision_0"]["selected"]},
            "key_builder": dict(LLM_KEY_BUILDER), "w": {"vision_0": 1.0, "robot_state": 0.0},
        }
        rows_llm.append({"arm": arm, "yaml": str(path), "sidecar": None})
        yaml_records[arm] = {"path": str(path), "sha256": sha256_of(path), "library": LLM_STEM,
                             "w": {"vision_0": 1.0, "robot_state": 0.0}}
        print(f"{arm:44s} single key (vision_0=1.0, robot_state=0.0)")

        matrix_pool = out_dir / f"matrix_{suite}_pool.yaml"
        matrix_llm = out_dir / f"matrix_{suite}_llm.yaml"
        matrix_pool.write_text(yaml.safe_dump({"suite": suite, "arms": rows_pool}, sort_keys=False))
        matrix_llm.write_text(yaml.safe_dump({"suite": suite, "arms": rows_llm}, sort_keys=False))
        apool = yaml.safe_load(pathlib.Path(APOOL_RECORD.format(suite=suite)).read_text())
        ref_dir = REFERENCE_DIR.format(suite=suite)
        manifest["suites"][suite] = {
            "apool_record": APOOL_RECORD.format(suite=suite),
            "apool_rollup_sha256": apool["rollup_sha256"],
            # Run sources are bound per run: the driver writes one
            # per_step.jsonl.launch.<run_id>.json per launch, and the summary
            # requires a checked record for every run that produced an accepted
            # episode. The list is filled in after the runs (one entry per
            # launch of the round). The reference arm predates run ids on the
            # launch side and keeps its frozen single record.
            "rounds": {
                "pool": {"matrix": str(matrix_pool), "arms": [r["arm"] for r in rows_pool],
                         "journal": str(args.data / "pool" / suite / "journal.jsonl"),
                         "per_step": str(args.data / "pool" / suite / "per_step.jsonl"),
                         "launch": [], "launch_binding": "run_id",
                         "launch_glob": str(args.data / "pool" / suite / "per_step.jsonl.launch.*.json")},
                "llm": {"matrix": str(matrix_llm), "arms": [r["arm"] for r in rows_llm],
                        "journal": str(args.data / "llm" / suite / "journal.jsonl"),
                        "per_step": str(args.data / "llm" / suite / "per_step.jsonl"),
                        "launch": [], "launch_binding": "run_id",
                        "launch_glob": str(args.data / "llm" / suite / "per_step.jsonl.launch.*.json")},
            },
            "reference": {"arms": [REFERENCE_ARM.format(suite=suite)],
                          "journal": f"{ref_dir}/journal.jsonl",
                          "per_step": f"{ref_dir}/per_step.jsonl",
                          "launch": f"{ref_dir}/per_step.jsonl.launch.json",
                          "launch_binding": "legacy",
                          "launch_note": "fusion-ablation run of 2026-09-14, launched before the driver "
                                         "recorded run_id on the launch side; frozen single record"},
            "libraries": libraries,
            "yamls": yaml_records,
        }
        all_ids += [r["arm"] for r in rows_pool + rows_llm]
        expect_pool = len(POOL_STEMS) - (1 if args.skip_clip else 0)
        if len(rows_pool) != expect_pool or len(rows_llm) != 1:
            raise SystemExit(f"{suite}: pool matrix has {len(rows_pool)} arms (expected {expect_pool}), "
                             f"llm matrix {len(rows_llm)} (expected 1)")
    if len(set(all_ids)) != len(all_ids):
        raise SystemExit(f"duplicate yaml_id across suites: {all_ids}")
    manifest["trials_per_task"] = 50
    manifest["episodes_per_arm"] = 500
    args.data.mkdir(parents=True, exist_ok=True)
    (args.data / "weights.json").write_text(json.dumps(weights_out, indent=1) + "\n")
    (args.data / "active_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(all_ids)} arms; weights -> {args.data / 'weights.json'}; manifest -> {args.data / 'active_manifest.json'}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", type=pathlib.Path, default=CFG, help="yaml output root")
    ap.add_argument("--data", type=pathlib.Path, default=DATA, help="fits / cells / weights / manifest root")
    ap.add_argument("--local-root", type=pathlib.Path, default=LOCAL_ART, help="artifact root on this host")
    ap.add_argument("--served-root", default=SERVED_ART, help="artifact root as the server sees it")
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("templates", help="fit templates + diagnostic cells; prints the fit commands")
    t.add_argument("--grid6-summary", type=pathlib.Path, default=GRID6_SUMMARY)
    f = sub.add_parser("final", help="arms, matrices, weights.json, active_manifest.json from finished fits")
    f.add_argument("--skip-clip", action="store_true", help="CLIP deferred by the preflight gate: 3 pool arms per suite")
    f.add_argument("--skip-clip-reason", default="", help="recorded in the manifest when --skip-clip is set")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "templates":
        run_templates(args)
    else:
        if args.skip_clip and not args.skip_clip_reason:
            raise SystemExit("--skip-clip needs --skip-clip-reason")
        run_final(args)


if __name__ == "__main__":
    main()
