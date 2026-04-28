"""Tests for ``phase1_spec.build_warmup_yaml`` + sibling-emit behaviour.

Validates the verdict_factor_judge B2.5 contract:

  1. ``build_warmup_yaml`` produces a deferred dump (path omitted, server
     fills) wrapped around ``always_warm_start@0.7``.
  2. The warmup yaml's ``dump.factors`` covers the SAME factor set the
     paired eval composite uses, so the dumped raw values feed the eval
     normalizer's keys exactly.
  3. The rendered warmup yaml passes ``load_cache_config`` even when
     ``/tmp/openpi_warmup`` does NOT exist on the host — this is the
     R3-G1R4 acceptance criterion that lets CI / offline tools validate
     the yamls without depending on the server's dump root.
  4. ``phase1_spec.main()`` emits one ``__warmup.yaml`` next to every eval
     yaml (24 cells × 2 = 48 files total).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from exp.verdict_factor_judge import phase1_spec
from exp.verdict_factor_judge.generate_yamls import write_yaml
from openpi.cache.config import load_cache_config


# ---------------------------------------------------------------------------
# build_warmup_yaml — pure dict construction
# ---------------------------------------------------------------------------


def test_build_warmup_yaml_emits_deferred_dump_with_omitted_path() -> None:
    """For a representative cell, the warmup yaml uses always_warm_start@0.7
    + DumpingJudge with deferred=True and an omitted path field."""
    cfg_id = "clip_w7_d4"
    stem = "f_full_d_all_t_dual_07"
    content = phase1_spec.build_warmup_yaml(cfg_id, stem)

    cp1 = content["checkpoints"]["cp1"]
    judge = cp1["judge"]
    assert judge["type"] == "always_warm_start"
    assert judge["start_t"] == 0.7

    dump = judge["dump"]
    assert dump["deferred"] is True
    # Path must be absent or empty — the server fills it from the warmup id.
    assert "path" not in dump or dump["path"] == ""

    expected_eval_id = f"{cfg_id}_phase1_{stem}"
    expected_warmup_id = f"{expected_eval_id}__warmup"
    assert dump["config_id"] == expected_warmup_id
    assert dump["config_id"].endswith("__warmup")


def test_build_warmup_yaml_dump_factors_match_eval_composite() -> None:
    """The warmup dump always over-collects across the full 5-factor set so
    one warmup run can serve every Phase 1 ablation variant within the cell.

    For the F-FULL eval (which itself uses all 5 factors), the dump.factors
    list deep-equals the eval composite's judge.factors list."""
    cfg_id = "clip_w7_d4"
    stem = "f_full_d_all_t_dual_07"

    eval_yaml = phase1_spec.build_yaml(cfg_id, stem)
    warmup_yaml = phase1_spec.build_warmup_yaml(cfg_id, stem)

    eval_factors = eval_yaml["checkpoints"]["cp1"]["judge"]["factors"]
    warmup_dump_factors = warmup_yaml["checkpoints"]["cp1"]["judge"]["dump"]["factors"]

    assert warmup_dump_factors == eval_factors


def test_build_warmup_yaml_widens_top_k_for_f2_extractor() -> None:
    """``DumpingJudge``'s F2 extractor needs the top-5 KNN candidates; the
    warmup yaml widens ``search_strategy.top_k`` to 5 even for cells whose
    eval composite only requires top_k=1."""
    # F1a-T-only eval has top_k=1; warmup must still be 5.
    warmup_yaml = phase1_spec.build_warmup_yaml(
        "clip_w7_d4", "f_f1a_t_only_d_all_t_full"
    )
    assert warmup_yaml["checkpoints"]["cp1"]["search_strategy"]["top_k"] == 5


# ---------------------------------------------------------------------------
# Offline validation — /tmp/openpi_warmup absent
# ---------------------------------------------------------------------------


def test_warmup_yaml_validates_when_warmup_root_absent(tmp_path: Path) -> None:
    """Render the warmup yaml to disk and load it via the public API.

    Skips when ``/tmp/openpi_warmup`` happens to exist (we only verify the
    deferred-validator gate; if the dir already exists the legacy path-check
    branch may also succeed and the test wouldn't actually pin our claim)."""
    if Path("/tmp/openpi_warmup").exists():
        pytest.skip(
            "/tmp/openpi_warmup exists on host; cannot exercise the "
            "deferred bypass branch in isolation"
        )

    cfg_id = "clip_w7_d4"
    stem = "f_min_cons_d_all_t_full"
    warmup_yaml_id = f"{cfg_id}_phase1_{stem}__warmup"
    yaml_path = tmp_path / f"{warmup_yaml_id}.yaml"
    write_yaml(yaml_path, phase1_spec.build_warmup_yaml(cfg_id, stem))

    # If load_cache_config raises here the deferred path is not honoured.
    cfg = load_cache_config(yaml_path)
    # Sanity: the loaded judge should still carry the deferred flag and
    # the empty path (server-side handler is what fills it; this offline
    # call doesn't go through that handler).
    cp1 = cfg.checkpoints["cp1"]
    assert cp1.judge.type == "always_warm_start"
    assert cp1.judge.dump is not None
    assert cp1.judge.dump.deferred is True
    assert cp1.judge.dump.path == ""


# ---------------------------------------------------------------------------
# main() — sibling emission
# ---------------------------------------------------------------------------


def test_phase1_spec_main_writes_sibling_warmup_for_every_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect the config root to ``tmp_path`` and assert that for each
    eval yaml a sibling ``__warmup.yaml`` lives in the same directory.

    Total expected file count: 3 cfg × 8 descriptors × 2 (eval + warmup)
    = 48 yaml files."""
    real_path = phase1_spec.Path

    def fake_path(arg: str) -> Path:
        # Only intercept the one literal path the spec uses to build the
        # config root — every other Path() call (e.g. via write_yaml) must
        # behave normally so file IO still works.
        if arg == "exp/verdict_factor_judge/config":
            return tmp_path
        return real_path(arg)

    monkeypatch.setattr(phase1_spec, "Path", fake_path)
    phase1_spec.main()

    # 3 cfg × 8 descriptors × 2 = 48
    all_yamls = sorted(tmp_path.rglob("*.yaml"))
    assert len(all_yamls) == 48, (
        f"Expected 48 yamls (24 eval + 24 warmup); got {len(all_yamls)}: "
        f"{[p.name for p in all_yamls]}"
    )

    # Every eval yaml must have a sibling __warmup.yaml in the same dir.
    eval_yamls = [p for p in all_yamls if not p.stem.endswith("__warmup")]
    warmup_yamls = [p for p in all_yamls if p.stem.endswith("__warmup")]
    assert len(eval_yamls) == 24
    assert len(warmup_yamls) == 24

    for eval_path in eval_yamls:
        sibling = eval_path.parent / f"{eval_path.stem}__warmup.yaml"
        assert sibling.exists(), f"Missing sibling for {eval_path}"

    # Sanity: verify one warmup file actually parses as yaml and carries
    # the expected dump.deferred=True flag (defends against template drift).
    sample = warmup_yamls[0]
    body = yaml.safe_load(sample.read_text())
    assert body["checkpoints"]["cp1"]["judge"]["type"] == "always_warm_start"
    assert body["checkpoints"]["cp1"]["judge"]["dump"]["deferred"] is True
