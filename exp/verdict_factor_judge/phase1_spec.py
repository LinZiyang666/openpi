"""Phase 1 yaml generator — single-factor ablation across 3 cfgs (24 yamls total).

Writes one yaml per (cfg_id, descriptor) cell of the 3 x 8 grid defined in
`logs/verdict_factor_judge_experiment_plan.log.md` §4 Phase 1. The eight
descriptors per cfg, in lex order (controls runner --runs index), are:

    1. f_f1a_t_only_d_all_t_full     - F1a-T only, all 4 desc, FULL_HIT
    2. f_f1b_a_only_d_all_t_full     - F1b-A only, W-MIX x 4 desc, FULL_HIT
    3. f_f1b_t_only_d_all_t_full     - F1b-T only, W-MIX x 4 desc, FULL_HIT
    4. f_full_d_all_t_dual_07        - F-FULL (5 factors), D-ALL, T-DUAL(0.7)
    5. f_min_a_d_all_t_full          - F-MIN-A (f1a_a), D-ALL, FULL_HIT
    6. f_min_a_d_dir_t_full          - F-MIN-A, D-DIR (single descriptor)
    7. f_min_a_d_jerk_t_full         - F-MIN-A, D-JERK (single descriptor)
    8. f_min_cons_d_all_t_full       - F-MIN-CONS (f2 only), FULL_HIT

Key design constants (plan-locked, see plan log for derivation):
- Composer = WeightedSum, uniform weights = 1.0, full_hit threshold = 0.5
- T-DUAL_07 yaml adds warm_start threshold 0.3 + warm_start_t 0.7
- all-NaN bootstrap fallback = WARM_START at start_t 0.7
- W-MIX windows = [(0,3) (1,1) (3,0) (0,5) (5,0)] - plan SS3.3 / SS3.3b B-3
- N-PCT directions for non_monotonic: curv_radius range:[0.3,0.7], cum_disp high
- F1a-A / F1a-T: window_k = 3 (matches Phase 0 dump factors)
- F2: K = 5 (matches Phase 0 dump factors)

This module is purely declarative + the file writes happen in `__main__`.
Re-running is idempotent; existing files at the target paths get overwritten.
"""

from __future__ import annotations

from pathlib import Path

from exp.verdict_factor_judge.generate_yamls import write_yaml

# ------------------------------------------------------------------
# CFG specs - one entry per cfg_id (matches plan SS3.7 lock)
# ------------------------------------------------------------------

CFG_SPECS = {
    "clip_w7_d4": {
        "subdir": "clip",
        "key_builder_type": "clip",
        "field_weights": {"vision_0": 0.1, "vision_1": 0.1, "prompt_emb": 0.0, "robot_state": 0.8},
        "vector_dims": {"vision_0": 512, "vision_1": 512, "prompt_emb": 2048, "robot_state": 32},
        "trajectory_depth": 4,
        "trajectory_weights": [0.4, 0.3, 0.2, 0.1],
        "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl",
    },
    "max_pool_w3_d5": {
        "subdir": "max_pool",
        "key_builder_type": "cp1_max_pool",
        "field_weights": {"vision_0": 0.5, "vision_1": 0.0, "prompt_emb": 0.0, "robot_state": 0.5},
        "vector_dims": {"vision_0": 2048, "vision_1": 2048, "prompt_emb": 2048, "robot_state": 32},
        "trajectory_depth": 5,
        "trajectory_weights": [0.35, 0.25, 0.2, 0.12, 0.08],
        "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/cp1_max_pool.pkl",
    },
    "spatial16_w8_d4": {
        "subdir": "spatial16",
        "key_builder_type": "cp1_spatial_pool_16",
        "field_weights": {"vision_0": 0.5, "vision_1": 0.25, "prompt_emb": 0.0, "robot_state": 0.25},
        "vector_dims": {"vision_0": 32768, "vision_1": 32768, "prompt_emb": 2048, "robot_state": 32},
        "trajectory_depth": 4,
        "trajectory_weights": [0.4, 0.3, 0.2, 0.1],
        "preload_pkl": "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
    },
}

# ------------------------------------------------------------------
# Factor descriptor sets - shared across cfgs
# ------------------------------------------------------------------

ALL_DESC = ["jerk", "dir", "curv_radius", "cum_disp"]
W_MIX = [
    {"past": 0, "future": 3},
    {"past": 1, "future": 1},
    {"past": 3, "future": 0},
    {"past": 0, "future": 5},
    {"past": 5, "future": 0},
]


def _f1a_factor(family: str, descriptors: list[str]) -> dict:
    return {"type": family, "params": {"window_k": 3, "descriptors": descriptors}}


def _f1b_factor(family: str, descriptors: list[str]) -> dict:
    return {
        "type": family,
        "params": {"windows": W_MIX, "descriptors": descriptors, "active_eps": 0.01},
    }


def _f2_factor() -> dict:
    return {"type": "f2", "params": {"K": 5}}


def _f1a_keys(family: str, descriptors: list[str]) -> list[str]:
    """Materialize key names for one F1a factor (single-window, no suffix)."""
    return [f"{family}_{d}" for d in descriptors]


def _f1b_keys(family: str, descriptors: list[str]) -> list[str]:
    """Materialize key names for one F1b factor (windowed, __pX_fY suffix)."""
    return [
        f"{family}_{d}__p{w['past']}_f{w['future']}"
        for d in descriptors
        for w in W_MIX
    ]


def _directions_for(family_keys: list[str]) -> dict[str, str]:
    """Return N-PCT directions block for the keys in `family_keys`.

    Only emits entries for non_monotonic descriptors (curv_radius, cum_disp);
    safe (`dir`) and risky (`jerk`) keys are handled implicitly by the
    composer's orientation-aware path. The validator only requires directions
    for non_monotonic keys whose weight is non-zero.
    """
    out: dict[str, str] = {}
    for k in family_keys:
        # Strip f1b's `__pX_fY` suffix to identify the descriptor.
        base = k.split("__")[0]
        if base.endswith("_curv_radius"):
            out[k] = "range:[0.3, 0.7]"
        elif base.endswith("_cum_disp"):
            out[k] = "high"
    return out


# ------------------------------------------------------------------
# Descriptor (= per-yaml) specs
# ------------------------------------------------------------------

# Each entry: descriptor_stem -> (factors_list, keys_list, tier_dual_07_or_full)
# tier flag: "FULL" or "DUAL_07"
DESCRIPTORS = {
    "f_f1a_t_only_d_all_t_full": (
        [_f1a_factor("f1a_t", ALL_DESC)], _f1a_keys("f1a_t", ALL_DESC), "FULL",
    ),
    "f_f1b_a_only_d_all_t_full": (
        [_f1b_factor("f1b_a", ALL_DESC)], _f1b_keys("f1b_a", ALL_DESC), "FULL",
    ),
    "f_f1b_t_only_d_all_t_full": (
        [_f1b_factor("f1b_t", ALL_DESC)], _f1b_keys("f1b_t", ALL_DESC), "FULL",
    ),
    "f_full_d_all_t_dual_07": (
        [
            _f1a_factor("f1a_a", ALL_DESC),
            _f1a_factor("f1a_t", ALL_DESC),
            _f1b_factor("f1b_a", ALL_DESC),
            _f1b_factor("f1b_t", ALL_DESC),
            _f2_factor(),
        ],
        (
            _f1a_keys("f1a_a", ALL_DESC)
            + _f1a_keys("f1a_t", ALL_DESC)
            + _f1b_keys("f1b_a", ALL_DESC)
            + _f1b_keys("f1b_t", ALL_DESC)
            + ["f2_var"]
        ),
        "DUAL_07",
    ),
    "f_min_a_d_all_t_full": (
        [_f1a_factor("f1a_a", ALL_DESC)], _f1a_keys("f1a_a", ALL_DESC), "FULL",
    ),
    "f_min_a_d_dir_t_full": (
        [_f1a_factor("f1a_a", ["dir"])], _f1a_keys("f1a_a", ["dir"]), "FULL",
    ),
    "f_min_a_d_jerk_t_full": (
        [_f1a_factor("f1a_a", ["jerk"])], _f1a_keys("f1a_a", ["jerk"]), "FULL",
    ),
    "f_min_cons_d_all_t_full": (
        [_f2_factor()], ["f2_var"], "FULL",
    ),
}


# ------------------------------------------------------------------
# Yaml builder
# ------------------------------------------------------------------


def _shared_search_strategy(cfg: dict) -> dict:
    return {
        "type": "weighted_rrf_knn",
        # Phase 1 uses CompositeJudge so min_top_k_hint auto-widens to F2's K=5
        # when the F-FULL / F-MIN-CONS yaml is loaded; setting top_k=1 here
        # keeps the wire shape minimal for F1a/F1b-only yamls (they have no
        # top_k requirement, so leaving it at 1 saves search work).
        "top_k": 1,
        "step_filter": "all",
        "field_similarity": {
            "vision_0": {"type": "cosine"},
            "vision_1": {"type": "cosine"},
            "prompt_emb": {"type": "cosine"},
            "robot_state": {
                "type": "l2",
                "to_similarity": {"type": "exp", "tau": 0.334717},
            },
        },
        "rrf_k": 60,
        "trajectory_depth": cfg["trajectory_depth"],
        "trajectory_weights": list(cfg["trajectory_weights"]),
    }


def _shared_keys_block(cfg: dict) -> dict:
    """Render the top-level `keys:` block per-cfg field_weights.

    vision_2 stays in the schema as a disabled placeholder (matches the
    warm_start template the cfgs were originally co-tuned with). Disabled
    keys carry zero weight and are skipped by the key_builder.
    """
    fw = cfg["field_weights"]
    return {
        "vision_0":    {"enabled": True,  "weight": fw["vision_0"]},
        "vision_1":    {"enabled": True,  "weight": fw["vision_1"]},
        "vision_2":    {"enabled": False, "weight": 0.0},
        "prompt_emb":  {"enabled": True,  "weight": fw["prompt_emb"]},
        "robot_state": {"enabled": True,  "weight": fw["robot_state"]},
    }


def _composer_block(keys: list[str], tier: str) -> dict:
    """WeightedSum composer with uniform weights + tier thresholds.

    - tier == "FULL"     -> only `full_hit: 0.5`
    - tier == "DUAL_07"  -> `full_hit: 0.5`, `warm_start: 0.3`, warm_start_t 0.7
      (validator SS5d requires warm_start strictly < full_hit; SS5a requires
      warm_start_t in CANONICAL_DENOISE_TIMESTEPS - 0.7 qualifies).
    """
    weights = {k: 1.0 for k in keys}
    directions = _directions_for(keys)
    composer: dict = {
        "type": "weighted_sum",
        "weights": weights,
        "tier_thresholds": {"full_hit": 0.5},
    }
    if directions:
        composer["directions"] = directions
    if tier == "DUAL_07":
        composer["tier_thresholds"]["warm_start"] = 0.3
        composer["warm_start_t"] = 0.7
    return composer


def build_warmup_yaml(cfg_id: str, descriptor_stem: str) -> dict:
    """Build the sibling warmup yaml dict for the (cfg_id, descriptor_stem) cell.

    The warmup yaml is loaded by ``run_phase.py`` BEFORE the matching eval yaml
    so that ``DumpingJudge`` can collect in-distribution factor raw values that
    later seed the eval normalizer's percentile buffer (plan B2 §3.4 / §3.5).

    Why the choices below:

    * ``judge.type = always_warm_start`` + ``start_t = 0.7`` — bypasses the
      composite judge so we never touch the (uninitialised) normalizer during
      warmup; every verdict short-circuits to WARM_START at t=0.7 just like
      the P0 fallback (this lets the policy actually advance and gives the
      dump factors meaningful per-step input).
    * ``judge.dump.factors`` = the SAME 5 factors as the eval composite uses
      so the dumped raw keys exactly match the keys the eval normalizer will
      preload from. (Phase 1 single-factor ablations also reuse this — the
      dump intentionally over-collects so warmup never has to be re-run when
      switching descriptor variants within the same cell.)
    * ``judge.dump.deferred = True`` + omitted ``path`` — the path is filled by
      the server's ``load_cache_config`` ctrl handler from
      ``<warmup_dump_root>/<warmup_yaml_id>.jsonl``; this lets
      ``validate_cache_config`` pass on CI / offline hosts where
      ``/tmp/openpi_warmup`` does not exist (plan R3-G1R4).
    * ``search_strategy.top_k = 5`` — the F2 dump extractor looks at the top-5
      KNN candidates' score variance; even though ``AlwaysWarmStartJudge``
      itself does not need search, we widen ``top_k`` so DumpingJudge's F2
      extractor sees the full neighborhood it expects.

    The returned dict carries ``dump.config_id`` set to the warmup id; the
    ``write_yaml`` invariant check pins ``config_id == file stem``, so the
    file the caller writes MUST be named ``<warmup_yaml_id>.yaml``.
    """
    cfg = CFG_SPECS[cfg_id]
    eval_yaml_id = f"{cfg_id}_phase1_{descriptor_stem}"
    warmup_yaml_id = f"{eval_yaml_id}__warmup"

    # Reuse the eval composite's full 5-factor list — DumpingJudge collects
    # raw per-step values across ALL factors regardless of whether the eval
    # yaml's composite weights any one of them, so over-collecting here is
    # intentional (lets us share one warmup dump across ablation siblings).
    full_factors = [
        _f1a_factor("f1a_a", ALL_DESC),
        _f1a_factor("f1a_t", ALL_DESC),
        _f1b_factor("f1b_a", ALL_DESC),
        _f1b_factor("f1b_t", ALL_DESC),
        _f2_factor(),
    ]

    judge = {
        "type": "always_warm_start",
        "start_t": 0.7,
        "dump": {
            # ``deferred=True`` tells the server's ``load_cache_config`` ctrl
            # handler to populate ``path`` from
            # ``<warmup_dump_root>/<config_id>.jsonl`` before validation runs;
            # we deliberately omit the ``path`` field entirely so its absence
            # is unambiguous (and the offline validator's `path == ""` branch
            # is exercised).
            "deferred": True,
            "config_id": warmup_yaml_id,
            "factors": full_factors,
        },
    }
    search_strategy = _shared_search_strategy(cfg)
    # F2 DumpExtractor inspects the top-K candidates' score spread; widen
    # ``top_k`` so the dump captures the same neighbourhood structure the
    # F-FULL eval composite expects (top_k=5).
    search_strategy["top_k"] = 5

    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": _shared_keys_block(cfg),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": judge,
                "search_strategy": search_strategy,
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {
                "preload_path": cfg["preload_pkl"],
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


def build_yaml(cfg_id: str, descriptor_stem: str) -> dict:
    cfg = CFG_SPECS[cfg_id]
    factors, keys, tier = DESCRIPTORS[descriptor_stem]
    judge = {
        "type": "composite",
        "factors": factors,
        "composer": _composer_block(keys, tier),
        "normalizer": {
            "type": "percentile_rolling",
            # window_size=50 (down from plan-default 200) is an empirical fix
            # for the multi-worker cold-start amplification observed on the
            # F-FULL T-DUAL_07 dead-loop run: with --num-workers 5, each
            # worker holds its own normalizer instance that needs window_size
            # valid samples per key before composer can run; the slowest F1b
            # window has ~44% NaN, so 200 / (1 - 0.44) ≈ 357 verdicts/worker
            # are cold (85% of the worker's ~420-verdict budget). Dropping to
            # 50 cuts cold-start to ~21% per worker (P0 fallback covers the
            # rest), at the cost of percentile resolution 0.5% -> 2%.
            # Re-evaluate in Phase 4 N-PCT-LEN ablation.
            "window_size": 50,
            "cold_start_strategy": "force_miss",
        },
        "all_nan_fallback": {
            "type": "warm_start",
            "start_t": 0.7,
        },
    }
    return {
        "enabled": True,
        "timer": {"enabled": False, "buffer_size": 10000, "output_csv_dir": None},
        "keys": _shared_keys_block(cfg),
        "key_builder": {"type": cfg["key_builder_type"]},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": judge,
                "search_strategy": _shared_search_strategy(cfg),
            },
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": dict(cfg["vector_dims"]),
            "in_memory": {
                "preload_path": cfg["preload_pkl"],
                "index_type": "brute_force",
            },
        },
        "write_policy": {"type": "never"},
    }


def main() -> None:
    config_root = Path("exp/verdict_factor_judge/config")
    written: list[Path] = []
    for cfg_id, cfg in CFG_SPECS.items():
        for stem in DESCRIPTORS:
            yaml_name = f"{cfg_id}_phase1_{stem}.yaml"
            # Phase-segregated layout: <cfg>/phase1/ keeps Phase 1 yamls
            # isolated from Phase 0 so client `--yaml-dir` points at one
            # phase only (no need for `--runs N-M` to skip cross-phase).
            phase_dir = config_root / cfg["subdir"] / "phase1"
            path = phase_dir / yaml_name
            content = build_yaml(cfg_id, stem)
            write_yaml(path, content)
            written.append(path)

            # B2.5 sibling: emit ``<eval_yaml_id>__warmup.yaml`` next to the
            # eval yaml so ``run_phase.py`` can load both from the same dir.
            # The naming contract (warmup id = eval id + "__warmup") is the
            # ONLY mechanism tying these two files together; renaming either
            # would silently break the runner.
            warmup_name = f"{cfg_id}_phase1_{stem}__warmup.yaml"
            warmup_path = phase_dir / warmup_name
            warmup_content = build_warmup_yaml(cfg_id, stem)
            write_yaml(warmup_path, warmup_content)
            written.append(warmup_path)
    print(f"Wrote {len(written)} Phase 1 yamls (eval + sibling warmup) under {config_root}/")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
