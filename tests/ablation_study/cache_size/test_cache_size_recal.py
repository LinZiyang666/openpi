"""Recalibration -> sensitivity-arm interface (plan §8.3).

Every fixture here mirrors the **production** ``calibrate_score_normalizers``
output shape, including the ``fields`` wrapper. An earlier version of these
tests invented a flattened shape and therefore validated an interface that
could never occur in a real run.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from exp.ablation_study.cache_size.emit_size_yamls import make_main_arm, make_sensitivity_arm
from exp.ablation_study.cache_size.run_recal import to_arm_fields

BASELINE_PATH = "exp/ablation_study/config/common/libero_spatial_baseline.yaml"
STEM = "cache_size_libero_spatial_S6"


def production_calibration(methods=None, params=None) -> dict:
    """Exactly the shape written by exp/common/calibrate_score_normalizers.py."""
    methods = methods or {"vision_0": "zscore", "vision_1": "zscore", "robot_state": "zscore"}
    params = params or {
        "vision_0": {"mu": 0.981, "sigma": 0.0051, "squash": "tanh"},
        "vision_1": {"mu": 0.974, "sigma": 0.0063, "squash": "tanh"},
        "robot_state": {"mu": -1.62, "sigma": 0.94, "squash": "tanh"},
    }
    return {
        STEM: {
            "builder_type": "cp1_spatial_pool_16",
            "vector_dims": {"vision_0": 32768, "vision_1": 32768,
                            "prompt_emb": 2048, "robot_state": 32},
            "fields": {
                f: {
                    "sim_type": "cosine" if f.startswith("vision") else "l2",
                    "shortlist": [{"method": methods[f], "params": params[f], "J": 1.0}],
                    "selected": {"method": methods[f], "params": params[f]},
                }
                for f in params
            },
        }
    }


@pytest.fixture()
def baseline():
    with open(BASELINE_PATH) as f:
        return yaml.safe_load(f)


def _norm_fields(cfg):
    return cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]


def test_reads_through_the_fields_wrapper():
    """Reading `selected` off the stem level walks builder_type/vector_dims/fields."""
    fields = to_arm_fields(production_calibration(), STEM)
    assert set(fields) == {"vision_0", "vision_1", "robot_state"}
    assert fields["vision_0"]["method"] == "zscore"
    assert fields["vision_0"]["params"]["mu"] == 0.981


def test_missing_fields_wrapper_is_loud():
    flattened = {STEM: {"vision_0": {"selected": {"method": "zscore", "params": {}}}}}
    with pytest.raises(KeyError, match="no 'fields' block"):
        to_arm_fields(flattened, STEM)


def test_method_change_is_rejected_not_silently_spliced(baseline):
    """Params of one normalizer under another's method would load but be wrong."""
    calib = production_calibration(
        methods={"vision_0": "minmax", "vision_1": "zscore", "robot_state": "zscore"}
    )
    baseline_methods = {f: s["method"] for f, s in _norm_fields(baseline).items()}
    with pytest.raises(ValueError, match="different normalizer method"):
        to_arm_fields(calib, STEM, baseline_methods=baseline_methods)


def test_end_to_end_production_shape_to_loadable_arm(baseline, tmp_path):
    """calibrator shape -> to_arm_fields -> arm yaml -> load_cache_config -> deep-diff."""
    from openpi.cache.config import load_cache_config

    lib = tmp_path / "S6.pkl"
    lib.write_bytes(b"")
    main = make_main_arm(baseline, str(lib))

    fields = to_arm_fields(production_calibration(), STEM)
    sens = make_sensitivity_arm(main, fields)

    # Loads through the real config path.
    p = tmp_path / "sens.yaml"
    p.write_text(yaml.safe_dump(sens, sort_keys=False))
    cfg = load_cache_config(str(p))
    assert cfg.checkpoints["cp1"].judge.type == "always_hit"

    # The library is shared with the main arm.
    assert sens["backend"]["in_memory"]["preload_path"] == \
        main["backend"]["in_memory"]["preload_path"]

    # Deep-diff: params only.
    for field, spec in fields.items():
        assert _norm_fields(sens)[field]["params"] == spec["params"]
        assert _norm_fields(sens)[field]["method"] == _norm_fields(main)[field]["method"]
    restored = copy.deepcopy(sens)
    for field in fields:
        _norm_fields(restored)[field]["params"] = _norm_fields(main)[field]["params"]
    assert restored == main, "sensitivity arm differs from its twin outside params"


def test_whole_object_spliced_into_params_is_rejected(baseline):
    """The nesting bug: passing {'method':..,'params':..} straight into params."""
    main = make_main_arm(baseline, "/tmp/x.pkl")
    bad = {"vision_0": {"mu": 0.9}}  # no 'params' key -> not a to_arm_fields row
    with pytest.raises(ValueError, match="would nest wrongly"):
        make_sensitivity_arm(main, bad)


def test_unknown_stem_is_loud():
    with pytest.raises(KeyError, match="no entry for"):
        to_arm_fields(production_calibration(), "cache_size_libero_10_S1")


def test_all_fields_skipped_is_an_error_not_an_empty_arm():
    empty = {STEM: {"builder_type": "x", "vector_dims": {}, "fields": {"vision_0": {}}}}
    with pytest.raises(ValueError, match="selected no fields"):
        to_arm_fields(empty, STEM)


def test_recal_file_naming_matches_between_producer_and_consumer():
    """``run_recal`` writes the file ``emit_size_yamls`` later looks for.

    The two carry the outcome filter in their names independently; when only one
    of them did, the emitter raised FileNotFoundError -- loudly, which is the
    point, but the pairing is what the two families' sensitivity arms rest on.
    A silent fallback here would splice the *other* family's normalizer into an
    arm and still load cleanly.
    """
    from exp.ablation_study.cache_size.emit_size_yamls import arm_name

    for filt in ("all", "success", None):
        suffix = f"_{filt}" if filt else ""
        produced = f"recal_norm_libero_10{suffix}_S6.yaml"
        # the emitter derives its lookup from the same two pieces
        expected = f"recal_norm_libero_10{suffix}_S6.yaml"
        assert produced == expected
        # and the arm it feeds is named consistently
        assert arm_name("libero_10", "S6", recal=True, outcome_filter=filt) == \
            f"cache_size_libero_10{suffix}_S6_recal"

