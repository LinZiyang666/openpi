"""YAML generator for the verdict factor judge experiment.

Enforces the cfg-prefixed naming invariant required by the Phase 5 outcome join:

    yaml stem == JudgeConfig.dump.config_id == cache_eval_results.json.config_id
                                                (runner uses ``run_id=yf.stem``)

The generator takes a per-phase spec (factor list, composer, normalizer, tier,
windows, descriptors) and emits one YAML per cfg subdirectory under
``exp/verdict_factor_judge/config/{cfg_id}/{cfg_id}_{phase_descriptor}.yaml``.

Public surface:

- ``check_yaml_invariant(yaml_path)`` — re-verify ``dump.config_id == yaml stem``
  for an existing YAML on disk; raises ``InvariantError`` on mismatch. Used by
  pre-commit / CI to detect manual edits that broke the invariant.
- ``write_yaml(yaml_path, content_dict)`` — write a YAML and validate the
  invariant before flushing; raises ``InvariantError`` if the content's
  ``dump.config_id`` would not match the file stem.

Coupling map:
  DEPENDS ON:  PyYAML (already a project dep)
  CONSUMED BY: phase-spec scripts (one per phase) that compose the YAML body
               and call write_yaml; pre-commit / CI invariant check.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class InvariantError(ValueError):
    """Raised when ``dump.config_id != yaml stem`` for a generated YAML."""


# ----------------------------------------------------------------------
# Public surface
# ----------------------------------------------------------------------


def expected_config_id(yaml_path: str | Path) -> str:
    """Return the canonical ``dump.config_id`` for a YAML at ``yaml_path``.

    By convention this is the file stem (filename without ``.yaml`` suffix).
    """
    return Path(yaml_path).stem


def extract_dump_config_id(content: dict) -> str | None:
    """Return ``checkpoints.cp1.judge.dump.config_id`` from a parsed YAML
    dict, or None if the path does not exist.

    Falls back to cp3 if cp1 is absent. Used as a single accessor so the
    invariant check stays robust to either checkpoint being the primary.
    """
    cps = content.get("checkpoints", {})
    for cp_name in ("cp1", "cp3"):
        cp = cps.get(cp_name) or {}
        judge = cp.get("judge") or {}
        dump = judge.get("dump") or {}
        cid = dump.get("config_id")
        if cid is not None:
            return str(cid)
    return None


def check_yaml_invariant(yaml_path: str | Path) -> None:
    """Read ``yaml_path`` and assert ``dump.config_id == yaml stem``.

    No-op when the YAML has no ``judge.dump`` block (non-dump YAMLs are
    exempt — only DumpingJudge-wrapping configs need the invariant).
    Raises ``InvariantError`` on mismatch.
    """
    path = Path(yaml_path)
    content = yaml.safe_load(path.read_text())
    if not isinstance(content, dict):
        raise InvariantError(f"{path}: YAML root is not a mapping")

    cid = extract_dump_config_id(content)
    if cid is None:
        return  # no dump block → invariant does not apply

    expected = expected_config_id(path)
    if cid != expected:
        raise InvariantError(
            f"{path}: dump.config_id={cid!r} does not match yaml stem={expected!r}; "
            f"the runner will produce cache_eval_results.json.config_id={expected!r} "
            f"and the join key (config_id, task_id, orig_init_state_idx) "
            f"will silently lose data."
        )


def write_yaml(yaml_path: str | Path, content: dict) -> None:
    """Validate the invariant on ``content``, then write to ``yaml_path``.

    If ``content`` carries a ``judge.dump`` block, its ``config_id`` is
    overwritten to the file stem before serialization (caller convenience —
    spec dicts can omit ``config_id`` entirely and the generator fills it in).
    Mismatches surface as ``InvariantError`` so phase scripts cannot
    accidentally produce un-joinable JSONL.
    """
    path = Path(yaml_path)
    expected = expected_config_id(path)

    cps = content.get("checkpoints", {})
    for cp in cps.values():
        judge = (cp or {}).get("judge") or {}
        dump = judge.get("dump")
        if dump is None:
            continue
        # Always pin to the stem — if the caller stored a different value
        # we surface the conflict explicitly rather than silently overwrite.
        provided = dump.get("config_id")
        if provided is not None and provided != expected:
            raise InvariantError(
                f"{path}: spec dump.config_id={provided!r} disagrees with yaml "
                f"stem={expected!r}. Either omit dump.config_id from the spec "
                f"(generator pins to stem) or rename the file."
            )
        dump["config_id"] = expected

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))
