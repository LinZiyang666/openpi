"""Tests for the ws2 YAML emitter: arm shapes, cell counts, real validation.

Every emitted config is run through the production loader + validator inside
the emitter itself, so these tests mainly pin the arm-shape invariants, the
manifest coupling of the control arm and the byte-reproducibility of a
re-emit.

The second half covers the pinned mode (pin plan §3-W6 / §6-S8): the
``expected_pin_id`` stamp and the per-teacher ``index_digest.json``. Those
tests tamper with a digest and then RECOMPUTE its summaries, so each one
proves the specific gate it names fires — a test that only broke the global
digest would pass with every per-cell check deleted.
"""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from exp.robocasa365 import emit_ws_search2_yamls as emitter
from exp.robocasa365.emit_ws_search2_yamls import (
    TEACHERS,
    build_cell,
    emit_arm,
    verify_cell,
)
from exp.robocasa365.emit_ws_search_yamls import weight_matrix

CALIB = {
    "builder_type": "cp1_groot_spatial_pool_16",
    "vector_dims": {"vision_0": 32768, "vision_1": 32768, "vision_2": 32768,
                    "prompt_emb": 2048, "robot_state": 20},
    "fields": {
        # zscore params mirror the real calibration json (mu/sigma/squash).
        f: {"sim_type": "cosine" if f.startswith("vision") else "l2",
            "selected": {"method": "zscore",
                         "params": {"mu": 0.85, "sigma": 0.03, "squash": "tanh"}}}
        for f in ("vision_0", "vision_1", "vision_2", "robot_state")
    },
}


def test_matrix_is_the_frozen_132_cell_set():
    configs = weight_matrix()
    assert len(configs) == 132
    assert all(abs(sum(w.values()) - 1.0) < 1e-9 for w in configs.values())


def test_main_arm_cell_carries_the_three_text_ivf_keys():
    weights = {"vision_2": 0.875, "robot_state": 0.125}
    cfg = build_cell(weights, CALIB, "groot_tp", text_ivf=True)
    assert cfg["checkpoints"]["cp1"]["search_strategy"]["type"] == "text_ivf_knn"
    assert cfg["backend"]["in_memory"]["index_type"] == "text_ivf"
    assert cfg["keys"]["prompt_emb"] == {"enabled": True, "weight": 0.0}
    assert cfg["backend"]["in_memory"]["preload_path"] == TEACHERS["groot_tp"]["preload"]
    # prompt_emb screens; it must never score.
    sn = cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]
    assert "prompt_emb" not in sn


def test_control_arm_cell_is_the_round1_shape_over_the_same_library():
    weights = {"vision_2": 0.875, "robot_state": 0.125}
    cfg = build_cell(weights, CALIB, "groot_tp", text_ivf=False)
    assert cfg["checkpoints"]["cp1"]["search_strategy"]["type"] == "weighted_score_sum_knn"
    assert cfg["backend"]["in_memory"]["index_type"] == "brute_force"
    assert cfg["keys"]["prompt_emb"]["enabled"] is False
    # Same library as the main arm: that is what makes the pair matched.
    assert cfg["backend"]["in_memory"]["preload_path"] == TEACHERS["groot_tp"]["preload"]


def test_both_arms_keep_the_pure_cache_recipe():
    for text_ivf in (True, False):
        cfg = build_cell({"vision_0": 1.0}, CALIB, "groot_tp", text_ivf=text_ivf)
        cp1 = cfg["checkpoints"]["cp1"]
        assert cp1["gate"]["type"] == "always_search"
        assert cp1["judge"]["type"] == "always_hit"
        assert cp1["search_strategy"]["top_k"] == 1
        assert cfg["write_policy"] == {"type": "never"}
        assert cfg["timer"]["enabled"] is False
        assert cfg["checkpoints"]["cp3"] == {"enabled": False,
                                             "search_strategy": {"type": "weighted_rrf_knn"}}


def test_verify_cell_rejects_a_half_converted_config():
    cfg = build_cell({"vision_0": 1.0}, CALIB, "groot_tp", text_ivf=True)
    cfg["backend"]["in_memory"]["index_type"] = "brute_force"
    with pytest.raises(AssertionError):
        verify_cell(cfg, "iso_vision_0", "groot_tp", text_ivf=True)


def test_emit_arm_writes_index_and_reproduces_byte_for_byte(tmp_path, monkeypatch):
    # The real validator needs the artifact only by path, never opens it, but
    # keep this unit hermetic: stub the on-disk validation.
    monkeypatch.setattr("exp.robocasa365.emit_ws_search2_yamls.validate_on_disk", lambda path: None)
    configs = weight_matrix()
    cids = sorted(configs)[:5]

    index = emit_arm(tmp_path / "main", cids, configs, CALIB, "groot_tp", text_ivf=True)
    assert set(index) == set(cids)
    first = {cid: (tmp_path / "main" / f"{cid}.yaml").read_text() for cid in cids}

    emit_arm(tmp_path / "main", cids, configs, CALIB, "groot_tp", text_ivf=True)
    assert {cid: (tmp_path / "main" / f"{cid}.yaml").read_text() for cid in cids} == first

    written = json.loads((tmp_path / "main" / "index.json").read_text())
    assert {c: written[c]["weights"] for c in cids} == {c: configs[c] for c in cids}
    loaded = yaml.safe_load(first[cids[0]])
    assert loaded["checkpoints"]["cp1"]["search_strategy"]["type"] == "text_ivf_knn"


def test_every_teacher_names_a_full704_library():
    for teacher, spec in TEACHERS.items():
        assert spec["stem"].endswith("_full704"), teacher
        assert "full704" in spec["preload"] and spec["preload"].startswith("/data/"), teacher


PI05_CALIB = {
    "builder_type": "cp1_spatial_pool_16",
    "vector_dims": {"vision_0": 32768, "vision_1": 32768, "vision_2": 32768,
                    "prompt_emb": 2048, "robot_state": 32},
    "fields": {
        f: {"sim_type": "cosine" if f.startswith("vision") else "l2",
            "selected": {"method": "zscore",
                         "params": {"mu": 0.85, "sigma": 0.03, "squash": "tanh"}}}
        for f in ("vision_0", "vision_1", "vision_2", "robot_state")
    },
}


def test_pi05_cells_carry_the_span_pooling_knobs():
    """The pi0.5 library was pooled over the instruction span; the config must say so.

    Its prompts embed a per-step state segment, so the library was built with
    masked + instruction-span pooling and records that in ``prompt_pool``. A
    config without the knobs would query a different pooling space, which the
    startup binding check refuses.
    """
    cfg = build_cell({"vision_2": 0.875, "robot_state": 0.125}, PI05_CALIB, "pi05", text_ivf=True)
    assert cfg["key_builder"] == {
        "type": "cp1_spatial_pool_16",
        "prompt_masked_pool": True,
        "prompt_instruction_span": True,
    }
    assert "pi05_spatial_pool_16_full704" in cfg["backend"]["in_memory"]["preload_path"]
    verify_cell(cfg, "iso_vision_2", "pi05", text_ivf=True)


def test_groot_cells_leave_the_knobs_off():
    """GR00T pools are not in the knob allowlist; its library records False/False."""
    cfg = build_cell({"vision_2": 1.0}, CALIB, "groot_tp", text_ivf=True)
    assert cfg["key_builder"] == {"type": "cp1_groot_spatial_pool_16"}
    verify_cell(cfg, "iso_vision_2", "groot_tp", text_ivf=True)


def test_verify_rejects_a_teacher_knob_mismatch():
    cfg = build_cell({"vision_2": 1.0}, PI05_CALIB, "pi05", text_ivf=True)
    cfg["key_builder"]["prompt_instruction_span"] = False
    with pytest.raises(AssertionError):
        verify_cell(cfg, "iso_vision_2", "pi05", text_ivf=True)


@pytest.mark.parametrize("teacher", ["groot_tp", "pi05"])
def test_both_teachers_pass_the_production_validator(tmp_path, teacher):
    from openpi.cache.config import load_cache_config, validate_cache_config

    calib = CALIB if teacher == "groot_tp" else PI05_CALIB
    cfg = build_cell({"vision_2": 0.875, "robot_state": 0.125}, calib, teacher, text_ivf=True)
    path = tmp_path / f"{teacher}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    validate_cache_config(load_cache_config(path))


@pytest.mark.parametrize("text_ivf", [True, False])
def test_both_arms_pass_the_production_validator(tmp_path, text_ivf):
    """The rule-6 change is what makes the main arm loadable at all.

    Not a stub: this is the loader the server runs, so a regression in the
    GR00T x text_ivf allowance fails here rather than at the first cell.
    """
    from openpi.cache.config import load_cache_config, validate_cache_config

    cfg = build_cell({"vision_2": 0.875, "robot_state": 0.125}, CALIB, "groot_tp", text_ivf=text_ivf)
    path = tmp_path / "cell.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    validate_cache_config(load_cache_config(path))


# ----------------------------------------------------------------------
# Pinned mode (pin plan §3-W6 / §6-S8): the PickPlace-only libraries, the
# expected_pin_id stamp, and the per-teacher index digest the eval driver
# re-checks before it dispatches.
# ----------------------------------------------------------------------

# Shape-valid but arbitrary: the emitter only ever copies this value through.
PIN_ID = "4d13ac5effce76c0ec253c9cef7dc2ed25dcb7d9ed4bae596e8b08ca483edcaa"


def pinned_calibration() -> dict:
    """A calibration json keyed the way the pinned re-calibration will be (by pkl stem)."""
    return {emitter.pinned_stem("groot_tp"): CALIB, emitter.pinned_stem("pi05"): PI05_CALIB}


@pytest.fixture(scope="module")
def pinned_tree(tmp_path_factory):
    """One emitted pinned tree (2 x 132 yaml + digest), shared by the digest tests.

    Emitted once: ``emit_pinned`` runs the production loader over all 264 files,
    which is the point of the fixture but not worth paying per test.
    """
    root = tmp_path_factory.mktemp("ws_search2_pnp")
    digest = emitter.emit_pinned(root, pinned_calibration(), PIN_ID)
    return root, digest


def test_pinned_spec_swaps_only_the_library():
    """Pinning changes which scenes the library holds, not how a key is built."""
    spec = emitter.pinned_teacher_spec("pi05", PIN_ID)
    assert spec["builder"] == TEACHERS["pi05"]["builder"]
    assert spec["knobs"] == TEACHERS["pi05"]["knobs"]
    assert spec["pin_id"] == PIN_ID
    # One tag names both the calibration key and the pkl, so they cannot drift.
    assert spec["stem"] == "pi05_spatial_pool_16_pnp_pinned"
    assert spec["preload"].endswith(f"/{spec['stem']}.pkl")
    assert spec["preload"] != TEACHERS["pi05"]["preload"]


def test_pinned_cell_stamps_the_expected_pin_id():
    spec = emitter.pinned_teacher_spec("groot_tp", PIN_ID)
    cfg = build_cell({"vision_2": 1.0}, CALIB, "groot_tp", text_ivf=True, spec=spec)
    assert cfg["backend"]["in_memory"]["expected_pin_id"] == PIN_ID
    assert cfg["backend"]["in_memory"]["preload_path"] == spec["preload"]
    verify_cell(cfg, "iso_vision_2", "groot_tp", text_ivf=True, spec=spec)


def test_unpinned_cell_carries_no_expectation():
    """Every pre-pinning library must keep loading: no key at all, not an empty one."""
    cfg = build_cell({"vision_2": 1.0}, CALIB, "groot_tp", text_ivf=True)
    assert "expected_pin_id" not in cfg["backend"]["in_memory"]


def test_verify_cell_rejects_a_wrong_or_missing_pin_id():
    spec = emitter.pinned_teacher_spec("groot_tp", PIN_ID)
    cfg = build_cell({"vision_2": 1.0}, CALIB, "groot_tp", text_ivf=True, spec=spec)
    cfg["backend"]["in_memory"]["expected_pin_id"] = "0" * 64
    with pytest.raises(AssertionError):
        verify_cell(cfg, "iso_vision_2", "groot_tp", text_ivf=True, spec=spec)
    del cfg["backend"]["in_memory"]["expected_pin_id"]
    with pytest.raises(AssertionError):
        verify_cell(cfg, "iso_vision_2", "groot_tp", text_ivf=True, spec=spec)


@pytest.mark.parametrize("teacher", ["groot_tp", "pi05"])
def test_pinned_cell_passes_the_production_validator(tmp_path, teacher):
    """expected_pin_id is a real field of InMemoryConfig, not a key the loader drops."""
    from openpi.cache.config import load_cache_config, validate_cache_config

    calib = CALIB if teacher == "groot_tp" else PI05_CALIB
    spec = emitter.pinned_teacher_spec(teacher, PIN_ID)
    cfg = build_cell({"vision_2": 0.875, "robot_state": 0.125}, calib, teacher,
                     text_ivf=True, spec=spec)
    path = tmp_path / f"{teacher}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    loaded = load_cache_config(path)
    validate_cache_config(loaded)
    assert loaded.backend.in_memory.expected_pin_id == PIN_ID


def test_pinned_tree_is_both_teachers_132_cells_each(pinned_tree):
    root, digest = pinned_tree
    assert sorted(digest["per_teacher"]) == sorted(TEACHERS)
    cids = set(weight_matrix())
    for teacher in TEACHERS:
        cells = digest["per_teacher"][teacher]["cells"]
        assert set(cells) == cids and len(cells) == 132
        arm = root / teacher / "main"
        assert len(list(arm.glob("*.yaml"))) == 132
        loaded = yaml.safe_load((arm / f"{sorted(cids)[0]}.yaml").read_text())
        assert loaded["backend"]["in_memory"]["expected_pin_id"] == PIN_ID


def test_digest_layering_keeps_the_two_teachers_apart(pinned_tree):
    """The two teachers share the cids; a flat {cid: sha} table would hide that."""
    _, digest = pinned_tree
    groot = digest["per_teacher"]["groot_tp"]["cells"]
    pi05 = digest["per_teacher"]["pi05"]["cells"]
    assert set(groot) == set(pi05)
    assert all(groot[cid] != pi05[cid] for cid in groot)
    assert digest["per_teacher"]["groot_tp"]["digest"] != digest["per_teacher"]["pi05"]["digest"]


def test_digest_schema_matches_the_frozen_shape(pinned_tree):
    root, digest = pinned_tree
    on_disk = json.loads((root / "index_digest.json").read_text())
    assert on_disk == digest
    assert sorted(on_disk) == ["global_digest", "per_teacher", "source_sha256"]
    assert sorted(on_disk["source_sha256"]) == [
        "emit_ws_search2_yamls.py", "emit_ws_search_yamls.py",
    ]
    for section in on_disk["per_teacher"].values():
        assert sorted(section) == ["cells", "digest"]


def test_verify_index_digest_accepts_a_freshly_emitted_tree(pinned_tree):
    root, digest = pinned_tree
    assert emitter.verify_index_digest(
        root, root / "index_digest.json", expected_pin_id=PIN_ID
    ) == digest


def test_verify_index_digest_binds_yaml_tree_to_runtime_manifest(pinned_tree):
    root, _ = pinned_tree
    with pytest.raises(ValueError, match="runtime manifest"):
        emitter.verify_index_digest(
            root,
            root / "index_digest.json",
            expected_pin_id="0" * 64,
        )


def test_pinned_config_dir_must_match_the_selected_teacher(pinned_tree):
    from exp.robocasa365.run_ws_search2 import assert_frozen_cell_set, pinned_config_root

    root, _ = pinned_tree
    assert pinned_config_root(root / "pi05" / "main", "pi05") == root.resolve()
    with pytest.raises(ValueError, match="groot_tp/main"):
        pinned_config_root(root / "pi05" / "main", "groot_tp")

    frozen = set(weight_matrix())
    assert_frozen_cell_set(sorted(frozen), frozen)
    with pytest.raises(ValueError, match="missing"):
        assert_frozen_cell_set(sorted(frozen)[1:], frozen)
    duplicate = sorted(frozen)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="missing"):
        assert_frozen_cell_set(duplicate, frozen)


def _rewritten_digest(tmp_path, per_teacher_cells, *, sources=None):
    """Write a fully self-consistent digest over tampered inputs.

    Recomputing the teacher and global digests is deliberate: it forces the
    check under test to be the one that fires, instead of letting the umbrella
    global-digest comparison catch everything.
    """
    path = tmp_path / "index_digest.json"
    path.write_text(json.dumps(emitter.build_index_digest(per_teacher_cells, sources=sources)))
    return path


def test_verify_index_digest_rejects_a_missing_cid(tmp_path, pinned_tree):
    root, digest = pinned_tree
    cells = {t: dict(s["cells"]) for t, s in digest["per_teacher"].items()}
    cells["pi05"].pop(sorted(cells["pi05"])[0])
    with pytest.raises(ValueError, match="cell set does not match"):
        emitter.verify_index_digest(root, _rewritten_digest(tmp_path, cells))


def test_verify_index_digest_rejects_an_extra_cid(tmp_path, pinned_tree):
    root, digest = pinned_tree
    cells = {t: dict(s["cells"]) for t, s in digest["per_teacher"].items()}
    cells["groot_tp"]["iso_vision_9"] = "0" * 64
    with pytest.raises(ValueError, match="cell set does not match"):
        emitter.verify_index_digest(root, _rewritten_digest(tmp_path, cells))


def test_verify_index_digest_rejects_an_edited_yaml(tmp_path, pinned_tree):
    """A hand-edited cell is the failure mode the per-file SHA exists for."""
    root, _ = pinned_tree
    tampered = tmp_path / "tree"
    shutil.copytree(root, tampered)
    cid = sorted(weight_matrix())[0]
    victim = tampered / "groot_tp" / "main" / f"{cid}.yaml"
    victim.write_text(victim.read_text().replace("top_k: 1", "top_k: 5"))
    with pytest.raises(ValueError, match="hashes to"):
        emitter.verify_index_digest(tampered, tampered / "index_digest.json")


def test_verify_index_digest_rejects_stale_source_sha256(tmp_path, pinned_tree):
    """An edit to weight_matrix() after the freeze must not pass unnoticed."""
    root, digest = pinned_tree
    cells = {t: dict(s["cells"]) for t, s in digest["per_teacher"].items()}
    sources = dict(digest["source_sha256"], **{"emit_ws_search_yamls.py": "0" * 64})
    path = _rewritten_digest(tmp_path, cells, sources=sources)
    with pytest.raises(ValueError, match="different emitter sources"):
        emitter.verify_index_digest(root, path)


def test_verify_index_digest_rejects_a_single_teacher_tree(tmp_path, pinned_tree):
    root, digest = pinned_tree
    cells = {"groot_tp": dict(digest["per_teacher"]["groot_tp"]["cells"])}
    with pytest.raises(ValueError, match="covers teachers"):
        emitter.verify_index_digest(root, _rewritten_digest(tmp_path, cells))


def test_verify_index_digest_rejects_a_forged_summary(tmp_path, pinned_tree):
    """Cells that check out one by one, summaries that were not recomputed."""
    root, digest = pinned_tree
    forged = json.loads(json.dumps(digest))
    forged["per_teacher"]["pi05"]["digest"] = "0" * 64
    path = tmp_path / "index_digest.json"
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="does not match"):
        emitter.verify_index_digest(root, path)

    forged["per_teacher"]["pi05"]["digest"] = digest["per_teacher"]["pi05"]["digest"]
    forged["global_digest"] = "0" * 64
    path.write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="global_digest"):
        emitter.verify_index_digest(root, path)


def test_verify_index_digest_rejects_a_missing_digest_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        emitter.verify_index_digest(tmp_path, tmp_path / "index_digest.json")


def test_pinned_emit_refuses_a_calibration_for_the_wrong_library(tmp_path):
    """The pinned libraries need their own normalizers; full704's stems must not pass."""
    stale = {TEACHERS[t]["stem"]: CALIB for t in TEACHERS}
    with pytest.raises(SystemExit, match="needs calibration stem"):
        emitter.emit_pinned(tmp_path, stale, PIN_ID)
