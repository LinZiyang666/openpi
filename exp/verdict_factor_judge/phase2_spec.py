"""Phase 2 Layer 1 yaml generator — within-factor descriptor / window ablation.

Mirrors phase1_spec.py but explores the inside of each factor:

- Layer 1.A — F1a-A close-out: curv_radius / cum_disp single + jerk+curv pair
  (Phase 1 left these gaps; jerk and dir single were already covered).
- Layer 1.B — F1a-T descriptor sweep: 4 single + 1 pair (jerk+dir).
  (Phase 1 only ran F1a-T all-4 — entire descriptor space unexplored.)
- Layer 1.C — F1b-A window x descriptor matrix: 4 window shapes (W-SHORT /
  W-PAST / W-FUT / W-SYM-S) x all-4-desc + W-SHORT x 3 single desc.
- Layer 1.D — F1b-T mirror of 1.C (same shapes, swap factor).
- Layer 1.E — W-LONG-RISK on F1b-A and F1b-T: large-window NaN-heavy probe
  (project memory: (5,5) ~48% NaN, (7,7) ~67% NaN). User-confirmed include.

Total: 26 yaml shapes per cfg x 3 cfg = 78 unique eval yamls + 78 sibling
warmup yamls = 156 yaml files.

Layout: each cfg's yamls split into ``phase2_layer1_a/`` (first 13 stems,
lex order) and ``phase2_layer1_b/`` (last 13). Combined with 3 cfgs that
gives 6 batch directories that map 1:1 to 6 parallel server slots:

    batch1 = clip       phase2_layer1_a       (clip server-A)
    batch2 = clip       phase2_layer1_b       (clip server-B)
    batch3 = max_pool   phase2_layer1_a       (max_pool server-A)
    batch4 = max_pool   phase2_layer1_b       (max_pool server-B)
    batch5 = spatial16  phase2_layer1_a       (spatial16 server-A)
    batch6 = spatial16  phase2_layer1_b       (spatial16 server-B)

Key design constants (inherited from phase1_spec unless noted):

- Composer = WeightedSum, uniform weights = 1.0, full_hit threshold = 0.5
- All Layer 1 yamls use T-FULL (no warm tier) — Layer 1's purpose is to find
  per-factor signal carriers; tier sweeps belong to Layer 2.B.
- all-NaN bootstrap fallback = WARM_START at start_t 0.7 (same as Phase 1).
- N-PCT directions for non_monotonic: curv_radius range:[0.3, 0.7], cum_disp high
- F1a-A / F1a-T: window_k = 3 (matches Phase 0 dump factors)
- F2: K = 5 (matches Phase 0 dump factors; not used in Layer 1 yamls)

The warmup yaml for every Phase 2 eval over-collects all 5 factors x all-4
descriptors x the union of all Phase 2 windows (W_MIX from Phase 1 plus the
new W_PAST / W_FUT / W_SYM_S / W_LONG_RISK windows). This way the warmup
dump is reusable: the eval normalizer always finds preload entries for the
keys it cares about, regardless of which descriptor / window subset the
eval yaml ablates down to.
"""

from __future__ import annotations

from pathlib import Path

from exp.verdict_factor_judge.generate_yamls import write_yaml
from exp.verdict_factor_judge.phase1_spec import CFG_SPECS

# ------------------------------------------------------------------
# Descriptors + windows
# ------------------------------------------------------------------

ALL_DESC = ["jerk", "dir", "curv_radius", "cum_disp"]

# Compact desc tokens for yaml stems (mirrors phase1's "d_jerk", "d_dir").
DESC_TOKEN = {
    "jerk": "jerk",
    "dir": "dir",
    "curv_radius": "curv",
    "cum_disp": "cum",
}

# Phase 1 baseline: W-MIX = (0,3)(1,1)(3,0)(0,5)(5,0).
W_MIX = [
    {"past": 0, "future": 3},
    {"past": 1, "future": 1},
    {"past": 3, "future": 0},
    {"past": 0, "future": 5},
    {"past": 5, "future": 0},
]

# Phase 2 Layer 1 windows.
W_SHORT = [
    {"past": 0, "future": 3},
    {"past": 1, "future": 1},
    {"past": 3, "future": 0},
]
W_PAST = [
    {"past": 3, "future": 0},
    {"past": 5, "future": 0},
]
W_FUT = [
    {"past": 0, "future": 3},
    {"past": 0, "future": 5},
]
W_SYM_S = [
    {"past": 1, "future": 1},
    {"past": 2, "future": 2},
    {"past": 3, "future": 3},
]
W_LONG_RISK = [
    {"past": 5, "future": 5},
    {"past": 7, "future": 7},
]

# Union of all windows used anywhere in Phase 2 — feeds the warmup dump so
# any eval yaml's keys already have preloaded buffer entries on first use.
# Order is deterministic (W_MIX first, then unique additions) so the dump
# JSONL columns line up across runs.
def _union_windows() -> list[dict]:
    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for source in (W_MIX, W_SHORT, W_PAST, W_FUT, W_SYM_S, W_LONG_RISK):
        for w in source:
            key = (w["past"], w["future"])
            if key not in seen:
                seen.add(key)
                out.append(dict(w))
    return out


W_UNION = _union_windows()


# ------------------------------------------------------------------
# Factor + key builders (mirror phase1_spec)
# ------------------------------------------------------------------


def _f1a_factor(family: str, descriptors: list[str]) -> dict:
    return {"type": family, "params": {"window_k": 3, "descriptors": descriptors}}


def _f1b_factor(family: str, descriptors: list[str], windows: list[dict]) -> dict:
    return {
        "type": family,
        "params": {"windows": windows, "descriptors": descriptors, "active_eps": 0.01},
    }


def _f2_factor() -> dict:
    return {"type": "f2", "params": {"K": 5}}


def _f1a_keys(family: str, descriptors: list[str]) -> list[str]:
    return [f"{family}_{d}" for d in descriptors]


def _f1b_keys(family: str, descriptors: list[str], windows: list[dict]) -> list[str]:
    return [
        f"{family}_{d}__p{w['past']}_f{w['future']}"
        for d in descriptors
        for w in windows
    ]


def _directions_for(family_keys: list[str]) -> dict[str, str]:
    """Same N-PCT directions block as phase1_spec — only emits non_monotonic
    entries (curv_radius / cum_disp); safe / risky descriptors handled by
    composer's orientation-aware path."""
    out: dict[str, str] = {}
    for k in family_keys:
        base = k.split("__")[0]
        if base.endswith("_curv_radius"):
            out[k] = "range:[0.3, 0.7]"
        elif base.endswith("_cum_disp"):
            out[k] = "high"
    return out


# ------------------------------------------------------------------
# Layer 1 descriptor table
# ------------------------------------------------------------------


def _build_descriptors() -> dict[str, tuple[list[dict], list[str], str]]:
    """Return {stem -> (factors, keys, tier)} for all Layer 1 yamls.

    Stems are returned in deterministic insertion order so [0:13] / [13:26]
    splits give reproducible phase2_layer1_a / phase2_layer1_b assignments.
    """
    out: dict[str, tuple[list[dict], list[str], str]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Layer 1.A — F1a-A close-out (3 yaml)
    # ──────────────────────────────────────────────────────────────────
    for d in ["curv_radius", "cum_disp"]:
        token = DESC_TOKEN[d]
        stem = f"f1a_a_d_{token}_only_t_full"
        out[stem] = (
            [_f1a_factor("f1a_a", [d])],
            _f1a_keys("f1a_a", [d]),
            "FULL",
        )
    pair = ["jerk", "curv_radius"]
    out["f1a_a_d_jerk_curv_pair_t_full"] = (
        [_f1a_factor("f1a_a", pair)],
        _f1a_keys("f1a_a", pair),
        "FULL",
    )

    # ──────────────────────────────────────────────────────────────────
    # Layer 1.B — F1a-T descriptor sweep (5 yaml)
    # ──────────────────────────────────────────────────────────────────
    for d in ALL_DESC:
        token = DESC_TOKEN[d]
        stem = f"f1a_t_d_{token}_only_t_full"
        out[stem] = (
            [_f1a_factor("f1a_t", [d])],
            _f1a_keys("f1a_t", [d]),
            "FULL",
        )
    pair = ["jerk", "dir"]
    out["f1a_t_d_jerk_dir_pair_t_full"] = (
        [_f1a_factor("f1a_t", pair)],
        _f1a_keys("f1a_t", pair),
        "FULL",
    )

    # ──────────────────────────────────────────────────────────────────
    # Layer 1.C — F1b-A windows x desc (7 yaml)
    # ──────────────────────────────────────────────────────────────────
    for win_name, win in [
        ("short", W_SHORT),
        ("past", W_PAST),
        ("fut", W_FUT),
        ("sym_s", W_SYM_S),
    ]:
        stem = f"f1b_a_w_{win_name}_d_all_t_full"
        out[stem] = (
            [_f1b_factor("f1b_a", ALL_DESC, win)],
            _f1b_keys("f1b_a", ALL_DESC, win),
            "FULL",
        )
    for d in ["jerk", "dir", "curv_radius"]:
        token = DESC_TOKEN[d]
        stem = f"f1b_a_w_short_d_{token}_t_full"
        out[stem] = (
            [_f1b_factor("f1b_a", [d], W_SHORT)],
            _f1b_keys("f1b_a", [d], W_SHORT),
            "FULL",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 1.D — F1b-T windows x desc (7 yaml, mirror of 1.C)
    # ──────────────────────────────────────────────────────────────────
    for win_name, win in [
        ("short", W_SHORT),
        ("past", W_PAST),
        ("fut", W_FUT),
        ("sym_s", W_SYM_S),
    ]:
        stem = f"f1b_t_w_{win_name}_d_all_t_full"
        out[stem] = (
            [_f1b_factor("f1b_t", ALL_DESC, win)],
            _f1b_keys("f1b_t", ALL_DESC, win),
            "FULL",
        )
    for d in ["jerk", "dir", "curv_radius"]:
        token = DESC_TOKEN[d]
        stem = f"f1b_t_w_short_d_{token}_t_full"
        out[stem] = (
            [_f1b_factor("f1b_t", [d], W_SHORT)],
            _f1b_keys("f1b_t", [d], W_SHORT),
            "FULL",
        )

    # ──────────────────────────────────────────────────────────────────
    # Layer 1.E — W-LONG-RISK probe on F1b-A and F1b-T (4 yaml)
    # ──────────────────────────────────────────────────────────────────
    for family in ["f1b_a", "f1b_t"]:
        out[f"{family}_w_long_risk_d_all_t_full"] = (
            [_f1b_factor(family, ALL_DESC, W_LONG_RISK)],
            _f1b_keys(family, ALL_DESC, W_LONG_RISK),
            "FULL",
        )
        out[f"{family}_w_long_risk_d_jerk_t_full"] = (
            [_f1b_factor(family, ["jerk"], W_LONG_RISK)],
            _f1b_keys(family, ["jerk"], W_LONG_RISK),
            "FULL",
        )

    return out


DESCRIPTORS = _build_descriptors()


# ------------------------------------------------------------------
# Yaml builder
# ------------------------------------------------------------------


def _shared_search_strategy(cfg: dict) -> dict:
    return {
        "type": "weighted_rrf_knn",
        # Eval yamls use top_k=1 (no F2 in Layer 1 yamls — all single-factor
        # F1a/F1b — so search has no min_top_k_hint requirement). The warmup
        # yaml widens this to 5 since DumpingJudge over-collects F2.
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
    fw = cfg["field_weights"]
    return {
        "vision_0":    {"enabled": True,  "weight": fw["vision_0"]},
        "vision_1":    {"enabled": True,  "weight": fw["vision_1"]},
        "vision_2":    {"enabled": False, "weight": 0.0},
        "prompt_emb":  {"enabled": True,  "weight": fw["prompt_emb"]},
        "robot_state": {"enabled": True,  "weight": fw["robot_state"]},
    }


def _composer_block(keys: list[str], tier: str) -> dict:
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
    """Build the sibling warmup yaml for one (cfg, stem) cell.

    The warmup judge stays AlwaysWarmStart @ 0.7 (same as Phase 1) so the
    policy advances naturally and DumpingJudge captures realistic factor
    distributions. The dump's factor list is the FULL Phase 2 superset:
    all 5 factors x all-4 desc x W_UNION (9 windows). This makes the dump
    reusable across all 26 Layer 1 eval yamls — none of them need their
    own warmup since each yaml's keys are a subset of the dumped keys.
    """
    cfg = CFG_SPECS[cfg_id]
    eval_yaml_id = f"{cfg_id}_phase2_{descriptor_stem}"
    warmup_yaml_id = f"{eval_yaml_id}__warmup"

    full_factors = [
        _f1a_factor("f1a_a", ALL_DESC),
        _f1a_factor("f1a_t", ALL_DESC),
        _f1b_factor("f1b_a", ALL_DESC, W_UNION),
        _f1b_factor("f1b_t", ALL_DESC, W_UNION),
        _f2_factor(),
    ]

    judge = {
        "type": "always_warm_start",
        "start_t": 0.7,
        "dump": {
            "deferred": True,
            "config_id": warmup_yaml_id,
            "factors": full_factors,
        },
    }
    search_strategy = _shared_search_strategy(cfg)
    # F2 DumpExtractor needs the top-K candidates' scores; widen to 5 so
    # the dump captures the same neighbourhood structure F2 would see in
    # an F-FULL eval yaml (matches phase1_spec convention).
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
            # Inherits Phase 1's window_size=50 fix for cold-start under
            # 5-worker parallelism. See phase1_spec.py for derivation.
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
    """Generate Layer 1 yamls + sibling warmups, split into a/b halves
    per cfg for 6-server (batch1-6) parallel scheduling."""
    config_root = Path("exp/verdict_factor_judge/config")
    stems = list(DESCRIPTORS.keys())  # insertion order — reproducible split
    half = len(stems) // 2  # 13 / 13 split for 26 stems
    half_a = stems[:half]
    half_b = stems[half:]

    written: list[Path] = []
    for cfg_id, cfg in CFG_SPECS.items():
        for half_name, half_stems in [("phase2_layer1_a", half_a), ("phase2_layer1_b", half_b)]:
            phase_dir = config_root / cfg["subdir"] / half_name
            for stem in half_stems:
                eval_name = f"{cfg_id}_phase2_{stem}.yaml"
                eval_path = phase_dir / eval_name
                write_yaml(eval_path, build_yaml(cfg_id, stem))
                written.append(eval_path)

                warmup_name = f"{cfg_id}_phase2_{stem}__warmup.yaml"
                warmup_path = phase_dir / warmup_name
                write_yaml(warmup_path, build_warmup_yaml(cfg_id, stem))
                written.append(warmup_path)

    print(f"Wrote {len(written)} Phase 2 Layer 1 yamls under {config_root}/")
    print(f"Per-cfg split: half_a={len(half_a)} stems, half_b={len(half_b)} stems")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
