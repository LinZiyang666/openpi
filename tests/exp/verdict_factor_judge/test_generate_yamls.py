"""Unit tests for verdict_factor_judge YAML generator invariant.

Covers:
  - check_yaml_invariant: matched -> no raise; mismatched -> raise
  - check_yaml_invariant: no-dump YAMLs are exempt
  - write_yaml: spec dump.config_id pinned to file stem on write
  - write_yaml: spec dump.config_id disagrees with stem -> raise
  - write_yaml: spec without dump block writes unchanged
"""

from __future__ import annotations

import pytest
import yaml

from exp.verdict_factor_judge.generate_yamls import (
    InvariantError,
    check_yaml_invariant,
    expected_config_id,
    write_yaml,
)


def _yaml_with_dump(stem_in_content: str) -> dict:
    return {
        "checkpoints": {
            "cp1": {
                "judge": {
                    "type": "always_hit",
                    "dump": {
                        "path": "/tmp/foo.jsonl",
                        "config_id": stem_in_content,
                        "factors": [{"type": "f2", "params": {"K": 5}}],
                    },
                },
            },
        },
    }


# ------------------------------------------------------------------
# expected_config_id
# ------------------------------------------------------------------


def test_expected_config_id_uses_stem(tmp_path):
    p = tmp_path / "clip_w7_d4_phase0_always_hit_dump.yaml"
    assert expected_config_id(p) == "clip_w7_d4_phase0_always_hit_dump"


# ------------------------------------------------------------------
# check_yaml_invariant
# ------------------------------------------------------------------


def test_invariant_passes_when_match(tmp_path):
    p = tmp_path / "clip_w7_d4_phase0_always_hit_dump.yaml"
    p.write_text(yaml.safe_dump(_yaml_with_dump("clip_w7_d4_phase0_always_hit_dump")))
    check_yaml_invariant(p)  # no raise


def test_invariant_raises_on_mismatch(tmp_path):
    p = tmp_path / "spatial16_w8_d4_phase0_always_hit_dump.yaml"
    p.write_text(yaml.safe_dump(_yaml_with_dump("clip_w7_d4")))  # wrong cid
    with pytest.raises(InvariantError, match="does not match"):
        check_yaml_invariant(p)


def test_invariant_skipped_when_no_dump_block(tmp_path):
    p = tmp_path / "anything.yaml"
    p.write_text(yaml.safe_dump({
        "checkpoints": {"cp1": {"judge": {"type": "always_hit"}}},
    }))
    check_yaml_invariant(p)  # no raise


# ------------------------------------------------------------------
# write_yaml
# ------------------------------------------------------------------


def test_write_pins_dump_config_id_to_stem(tmp_path):
    p = tmp_path / "max_pool_w3_d5_phase1_f2_only.yaml"
    spec = _yaml_with_dump("placeholder_will_be_overwritten")
    # Drop the existing config_id so caller doesn't have to manage it
    spec["checkpoints"]["cp1"]["judge"]["dump"].pop("config_id")
    write_yaml(p, spec)

    written = yaml.safe_load(p.read_text())
    assert (
        written["checkpoints"]["cp1"]["judge"]["dump"]["config_id"]
        == "max_pool_w3_d5_phase1_f2_only"
    )


def test_write_raises_when_spec_disagrees_with_stem(tmp_path):
    p = tmp_path / "max_pool_w3_d5_phase0_always_hit_dump.yaml"
    spec = _yaml_with_dump("clip_w7_d4_phase0_always_hit_dump")  # different
    with pytest.raises(InvariantError, match="disagrees with yaml stem"):
        write_yaml(p, spec)
    # File should not exist on raise
    assert not p.exists()


def test_write_no_dump_block_passes_through(tmp_path):
    p = tmp_path / "anything.yaml"
    spec = {"checkpoints": {"cp1": {"judge": {"type": "always_hit"}}}}
    write_yaml(p, spec)
    assert p.exists()


def test_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "deep" / "clip_w7_d4_phase0.yaml"
    spec = _yaml_with_dump("clip_w7_d4_phase0")
    write_yaml(p, spec)
    assert p.exists()
