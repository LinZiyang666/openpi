"""Emit ws2 (text-IVF round) search YAMLs over the full704 bucketed library.

Two arms out of one weight matrix (plan §2-D4 / §3-W2):

- **main** (all 132 cells): the round-1 recipe with three text-IVF keys fixed
  up on the built dict — ``search_strategy.type -> text_ivf_knn``,
  ``backend.in_memory.index_type -> text_ivf`` and
  ``keys.prompt_emb -> {enabled: true, weight: 0}`` (the validator's screening
  requirement; the bucket replaces task scoping, prompt_emb never scores).
- **control** (the 12 manifest ws2c cells only): the round-1 shape verbatim
  (``weighted_score_sum_knn`` + task filter + brute_force) over the SAME
  full704 preload — the matched arm that separates library growth from bucket
  scoping.

A third mode, ``--pinned-objects``, re-emits the SAME 132-cell main arm over
the PickPlace-only pinned libraries (pin plan §3-W6): same builder, same
pooling knobs, only the preload swapped, plus
``backend.in_memory.expected_pin_id`` so a server handed the wrong library
refuses to start instead of serving plausible numbers drawn from a different
scene distribution. That mode emits both teachers in one invocation and seals
the tree into ``index_digest.json``, which the eval driver re-checks before it
dispatches (``verify_index_digest``). The digest is layered per teacher: the
two teachers share the 132 cids but not the yaml text, so a flat ``{cid: sha}``
table would let one teacher silently overwrite the other and still count 132.

Reuses ``exp.weighted_sum.emit_yamls.build_eval_config`` and the round-1
``weight_matrix``/cp3-pin unchanged; the round-1 emitter and its outputs are
not touched. Every emitted yaml is re-loaded through ``load_cache_config`` +
``validate_cache_config`` before the index is written (fail-fast, no partial
trees), and shape invariants are asserted per arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from exp.robocasa365 import emit_ws_search_yamls
from exp.robocasa365.emit_ws_search_yamls import weight_matrix
from exp.robocasa365.pinned_objects import canonical_json, load_pin_manifest
from exp.weighted_sum.emit_yamls import build_eval_config

# Per-teacher deployment facts. ``knobs`` must equal the artifact's recorded
# ``prompt_pool`` metadata or the startup binding check refuses the config:
# the pi0.5 library was pooled over the instruction span (its prompts carry a
# state segment that would otherwise drift every step), the GR00T one was not
# (its prompt embeddings are unpadded and already bit-stable within a task).
TEACHERS = {
    "groot_tp": {
        "builder": "cp1_groot_spatial_pool_16",
        "stem": "groot_tp_spatial_pool_16_full704",
        # The servers resolve this on weilandserver, where /data is local.
        "preload": "/data/robocasa365_cache/cache_artifacts_text_ivf/"
                   "groot_tp_spatial_pool_16_full704.pkl",
        "knobs": {},
    },
    "pi05": {
        "builder": "cp1_spatial_pool_16",
        "stem": "pi05_spatial_pool_16_full704",
        "preload": "/data/robocasa365_cache/cache_artifacts_text_ivf/"
                   "pi05_spatial_pool_16_full704.pkl",
        "knobs": {"prompt_masked_pool": True, "prompt_instruction_span": True},
    },
}

# Round-1's real-load-proven cp3 pin (see emit_ws_search_yamls docstring for
# the exact trap it guards against).
CP3_PIN = {"enabled": False, "search_strategy": {"type": "weighted_rrf_knn"}}

DEFAULT_OUT_ROOT = "exp/robocasa365/config/ws_search2"
# The pinned round writes its own tree; round-2's outputs stay byte-identical.
PINNED_OUT_ROOT = "exp/robocasa365/config/ws_search2_pnp"
DIGEST_NAME = "index_digest.json"
MAIN_ARM = "main"

# The pinned libraries are a W6 deliverable and do not exist on disk yet, so
# both the path and the calibration key are derived from ONE tag: the
# calibration json is keyed by pkl stem (calibrate_score_normalizers.py:213),
# and deriving both from the same string is what keeps a rename from pointing
# the config at one library while reading another one's normalizers.
PINNED_LIBRARY_TAG = "pnp_pinned"
DEFAULT_PINNED_PRELOAD_DIR = "/data/robocasa365_cache/cache_artifacts_pnp_pinned"
_FULL704_SUFFIX = "_full704"

# Domain separators, for the same reason ``pinned_objects`` has them: a
# one-teacher table would otherwise hash identically as a teacher section and
# as the whole table, and the two identities are only worth having if they are
# independent.
_TEACHER_DIGEST_DOMAIN = "robocasa365/ws2_index_teacher/v1"
_GLOBAL_DIGEST_DOMAIN = "robocasa365/ws2_index/v1"


def pinned_stem(teacher: str, library_tag: str = PINNED_LIBRARY_TAG) -> str:
    """Calibration stem (== pkl stem) of a teacher's pinned library."""
    base = TEACHERS[teacher]["stem"]
    if not base.endswith(_FULL704_SUFFIX):
        raise ValueError(f"teacher {teacher!r} stem {base!r} is not a full704 stem")
    return f"{base[: -len(_FULL704_SUFFIX)]}_{library_tag}"


def pinned_teacher_spec(
    teacher: str,
    pin_id: str,
    *,
    preload_dir: str = DEFAULT_PINNED_PRELOAD_DIR,
    library_tag: str = PINNED_LIBRARY_TAG,
) -> dict:
    """Round-2's teacher spec with the library swapped for the pinned one.

    Builder and pooling knobs are copied on purpose: pinning changes which
    scenes the library describes, not how a key is built, so a knob that drifted
    here would be refused by the startup binding check for the wrong reason.
    """
    stem = pinned_stem(teacher, library_tag)
    return {
        **TEACHERS[teacher],
        "stem": stem,
        "preload": f"{preload_dir.rstrip('/')}/{stem}.pkl",
        "pin_id": pin_id,
    }


def build_cell(
    weights: dict[str, float], calib_entry: dict, teacher: str, *, text_ivf: bool,
    spec: dict | None = None,
) -> dict:
    """Build one cell config: the round-1 recipe, plus the text-IVF keys for the main arm.

    ``spec`` overrides the teacher's library facts (pinned mode); it carries the
    same keys as a ``TEACHERS`` entry plus an optional ``pin_id``.
    """
    spec = TEACHERS[teacher] if spec is None else spec
    cfg = build_eval_config(
        builder_type=calib_entry["builder_type"],
        vector_dims=calib_entry["vector_dims"],
        preload_path=spec["preload"],
        weights=weights,
        fields_calib=calib_entry["fields"],
    )
    cfg["checkpoints"]["cp3"] = dict(CP3_PIN)
    cfg["key_builder"].update(spec["knobs"])
    if text_ivf:
        cfg["checkpoints"]["cp1"]["search_strategy"]["type"] = "text_ivf_knn"
        cfg["backend"]["in_memory"]["index_type"] = "text_ivf"
        cfg["keys"]["prompt_emb"] = {"enabled": True, "weight": 0.0}
    if spec.get("pin_id"):
        # Enforced at load by config.py:_check_pin_identity_binding, which
        # refuses an artifact built under a different pin table (or under none).
        cfg["backend"]["in_memory"]["expected_pin_id"] = spec["pin_id"]
    return cfg


def verify_cell(
    cfg: dict, cid: str, teacher: str, *, text_ivf: bool, spec: dict | None = None
) -> None:
    """Arm-shape invariants + the real validator on a round-tripped load."""
    ss = cfg["checkpoints"]["cp1"]["search_strategy"]
    weights = {f: k["weight"] for f, k in cfg["keys"].items() if k.get("enabled")}
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"{cid}: enabled weights must sum to 1"
    if text_ivf:
        assert ss["type"] == "text_ivf_knn", cid
        assert cfg["backend"]["in_memory"]["index_type"] == "text_ivf", cid
        assert cfg["keys"]["prompt_emb"] == {"enabled": True, "weight": 0.0}, cid
    else:
        assert ss["type"] == "weighted_score_sum_knn", cid
        assert cfg["backend"]["in_memory"]["index_type"] == "brute_force", cid
        assert not cfg["keys"]["prompt_emb"]["enabled"], cid
    assert cfg["write_policy"] == {"type": "never"}, cid
    assert cfg["backend"]["vector_dims"]["prompt_emb"] == 2048, cid
    spec = TEACHERS[teacher] if spec is None else spec
    assert cfg["key_builder"]["type"] == spec["builder"], cid
    # Knobs must match the artifact's prompt_pool metadata exactly; the startup
    # binding check compares them and refuses a mismatch in either direction.
    for knob in ("prompt_masked_pool", "prompt_instruction_span"):
        assert cfg["key_builder"].get(knob, False) == spec["knobs"].get(knob, False), (cid, knob)
    assert cfg["backend"]["in_memory"]["preload_path"] == spec["preload"], cid
    # Absent and None are the same thing here: an unpinned arm must carry no
    # expectation at all, a pinned one exactly the manifest's identity.
    assert cfg["backend"]["in_memory"].get("expected_pin_id") == spec.get("pin_id"), cid


def validate_on_disk(path: Path) -> None:
    """Load and validate an emitted YAML through the production loader."""
    from openpi.cache.config import load_cache_config, validate_cache_config

    validate_cache_config(load_cache_config(path))


def emit_arm(
    out_dir: Path, cids: list[str], configs: dict, calib_entry: dict, teacher: str,
    *, text_ivf: bool, spec: dict | None = None,
) -> dict:
    """Emit one arm's YAMLs plus its index.json, validating every file as it lands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for cid in cids:
        weights = configs[cid]
        cfg = build_cell(weights, calib_entry, teacher, text_ivf=text_ivf, spec=spec)
        verify_cell(cfg, cid, teacher, text_ivf=text_ivf, spec=spec)
        path = out_dir / f"{cid}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        validate_on_disk(path)
        index[cid] = {"file": path.name, "weights": weights}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    return index


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_sha256() -> dict[str, str]:
    """Hash the two files the cell set and the yaml shape actually come from.

    ``weight_matrix`` lives in the round-1 emitter, so an edit there silently
    changes what "the 132 cells" are; recording both sources is what makes that
    edit visible to a run that was frozen before it.
    """
    paths = (Path(emit_ws_search_yamls.__file__), Path(__file__))
    return {p.name: _sha256_text(p.read_text()) for p in paths}


def teacher_digest(teacher: str, cells: dict[str, str]) -> str:
    """Summary hash of one teacher's ``{cid: sha256(yaml_text)}`` table."""
    payload = {"domain": _TEACHER_DIGEST_DOMAIN, "teacher": teacher, "cells": cells}
    return _sha256_text(canonical_json(payload))


def build_index_digest(
    per_teacher_cells: dict[str, dict[str, str]], sources: dict[str, str] | None = None
) -> dict:
    """The frozen summary of an emitted pinned tree (pin plan §3-W6 schema).

    ``sources`` defaults to the live emitter sources; it is a parameter only so
    a recomputation can be checked against a digest's own recorded values
    instead of against whatever the files say right now.
    """
    per_teacher = {
        teacher: {"cells": cells, "digest": teacher_digest(teacher, cells)}
        for teacher, cells in sorted(per_teacher_cells.items())
    }
    sources = source_sha256() if sources is None else sources
    payload = {
        "domain": _GLOBAL_DIGEST_DOMAIN,
        "per_teacher": per_teacher,
        "source_sha256": sources,
    }
    return {
        "per_teacher": per_teacher,
        "source_sha256": sources,
        "global_digest": _sha256_text(canonical_json(payload)),
    }


def verify_index_digest(
    config_root: Path,
    digest_path: Path,
    *,
    expected_pin_id: str | None = None,
) -> dict:
    """Prove an emitted pinned tree is still the tree the digest froze (§6-S8).

    The eval driver calls this before it dispatches anything: a drifted cell
    set, a hand-edited yaml, or an emitter edit after the freeze all mean the
    run would not be the run the digest describes, and every one of them is
    invisible in the results.

    Raises:
        ValueError: on the first inconsistency, naming it. Mirrors
            ``pinned_objects.load_pin_manifest`` — this is a library check run
            inside another driver, not a CLI argument error.
    """
    config_root = Path(config_root)
    digest_path = Path(digest_path)
    try:
        doc = json.loads(digest_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"index digest {digest_path} does not exist") from exc

    absent = sorted({"per_teacher", "source_sha256", "global_digest"} - set(doc))
    if absent:
        raise ValueError(f"index digest {digest_path} is missing {absent}")

    live_sources = source_sha256()
    if doc["source_sha256"] != live_sources:
        raise ValueError(
            f"index digest {digest_path} was written by different emitter sources: "
            f"recorded {doc['source_sha256']}, current {live_sources}"
        )

    per_teacher = doc["per_teacher"]
    if set(per_teacher) != set(TEACHERS):
        raise ValueError(
            f"index digest {digest_path} covers teachers {sorted(per_teacher)}, "
            f"expected {sorted(TEACHERS)}"
        )

    expected_cids = set(weight_matrix())
    for teacher in sorted(per_teacher):
        section = per_teacher[teacher]
        cells = section.get("cells")
        if not isinstance(cells, dict):
            raise ValueError(f"{teacher}: index digest section has no cells table")
        missing = sorted(expected_cids - set(cells))
        extra = sorted(set(cells) - expected_cids)
        if missing or extra:
            raise ValueError(
                f"{teacher}: cell set does not match weight_matrix(); "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        if len(cells) != 132:
            raise ValueError(f"{teacher}: expected 132 cells, got {len(cells)}")
        arm_dir = config_root / teacher / MAIN_ARM
        for cid in sorted(cells):
            path = arm_dir / f"{cid}.yaml"
            try:
                yaml_text = path.read_text()
                actual = _sha256_text(yaml_text)
            except FileNotFoundError as exc:
                raise ValueError(f"{teacher}: {path} is missing") from exc
            if actual != cells[cid]:
                raise ValueError(
                    f"{teacher}: {path} hashes to {actual}, digest says {cells[cid]}"
                )
            if expected_pin_id is not None:
                cfg = yaml.safe_load(yaml_text)
                if not isinstance(cfg, dict):
                    raise ValueError(f"{teacher}: {path} does not contain a config mapping")
                actual_pin_id = cfg.get("backend", {}).get("in_memory", {}).get("expected_pin_id")
                if actual_pin_id != expected_pin_id:
                    raise ValueError(
                        f"{teacher}: {path} expects pin_id {actual_pin_id!r}, "
                        f"but the runtime manifest hashes to {expected_pin_id!r}"
                    )
        recomputed = teacher_digest(teacher, cells)
        if section.get("digest") != recomputed:
            raise ValueError(
                f"{teacher}: recorded digest {section.get('digest')!r} does not match "
                f"its own cells table ({recomputed})"
            )

    rebuilt = build_index_digest(
        {t: per_teacher[t]["cells"] for t in per_teacher}, sources=doc["source_sha256"]
    )
    if rebuilt["global_digest"] != doc["global_digest"]:
        raise ValueError(
            f"index digest {digest_path} records global_digest {doc['global_digest']!r} "
            f"but its contents hash to {rebuilt['global_digest']}"
        )
    return doc


def calibration_entry(calib: dict, spec: dict, teacher: str) -> dict:
    """The teacher's slice of a Phase-1 calibration json, builder-checked."""
    if spec["stem"] not in calib:
        raise SystemExit(
            f"teacher {teacher!r} needs calibration stem {spec['stem']!r}, "
            f"but the json has {sorted(calib)}"
        )
    entry = calib[spec["stem"]]
    if entry["builder_type"] != spec["builder"]:
        raise SystemExit(
            f"calibration was computed for builder {entry['builder_type']!r}, "
            f"but teacher {teacher!r} runs {spec['builder']!r}"
        )
    return entry


def emit_pinned(
    out_root: Path,
    calib: dict,
    pin_id: str,
    *,
    preload_dir: str = DEFAULT_PINNED_PRELOAD_DIR,
    library_tag: str = PINNED_LIBRARY_TAG,
) -> dict:
    """Emit both teachers' pinned main arms and seal the tree into one digest.

    Both teachers in one invocation on purpose: the digest is only worth having
    if it covers the whole frozen tree, and a per-teacher invocation would have
    to read the other teacher's section back in to rewrite the global digest.
    The written digest is verified before returning — the tree is either
    completely frozen or the emit fails.
    """
    out_root = Path(out_root)
    configs = weight_matrix()
    cids = sorted(configs)
    per_teacher_cells: dict[str, dict[str, str]] = {}
    for teacher in sorted(TEACHERS):
        spec = pinned_teacher_spec(
            teacher, pin_id, preload_dir=preload_dir, library_tag=library_tag
        )
        calib_entry = calibration_entry(calib, spec, teacher)
        arm_dir = out_root / teacher / MAIN_ARM
        index = emit_arm(arm_dir, cids, configs, calib_entry, teacher,
                         text_ivf=True, spec=spec)
        assert len(index) == 132, f"{teacher}: main arm must be 132 cells, got {len(index)}"
        # Hash what actually landed on disk, not what we meant to write.
        per_teacher_cells[teacher] = {
            cid: _sha256_text((arm_dir / f"{cid}.yaml").read_text()) for cid in cids
        }
    digest = build_index_digest(per_teacher_cells)
    digest_path = out_root / DIGEST_NAME
    digest_path.write_text(json.dumps(digest, indent=1, sort_keys=True))
    verify_index_digest(out_root, digest_path, expected_pin_id=pin_id)
    return digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", required=True,
                    help="Phase-1 calibration json of the library being emitted: the "
                         "full704 one for the round-2 arms, W6's re-calibration for "
                         "--pinned-objects (round-1's n5 json is stale for both)")
    ap.add_argument("--manifest",
                    help="selection_manifest.json with the ws2c segment (control-arm "
                         "cells); required unless --pinned-objects is given")
    ap.add_argument("--out-root",
                    help=f"default {DEFAULT_OUT_ROOT}, or {PINNED_OUT_ROOT} in pinned mode")
    ap.add_argument("--teacher", choices=sorted(TEACHERS),
                    help="round-2 arms only (default groot_tp); pinned mode always "
                         "emits both teachers")
    ap.add_argument("--pinned-objects",
                    help="pin table (exp/robocasa365/config/pnp_pinned_objects.json). "
                         "Switches to pinned mode: both teachers, main arm only, "
                         "expected_pin_id on every cell, index_digest.json written")
    ap.add_argument("--pinned-preload-dir", default=DEFAULT_PINNED_PRELOAD_DIR,
                    help="directory the servers resolve the pinned pkls in. The "
                         "libraries themselves are a W6 deliverable and need not "
                         "exist when the yamls are emitted")
    ap.add_argument("--pinned-library-tag", default=PINNED_LIBRARY_TAG,
                    help="library tag; names BOTH the pkl file and the calibration "
                         "stem it is looked up under")
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text())

    if args.pinned_objects:
        if args.teacher:
            raise SystemExit(
                "--teacher is not accepted with --pinned-objects: the digest is only "
                "meaningful over a tree that holds both teachers"
            )
        pin_id, _ = load_pin_manifest(args.pinned_objects)
        out_root = Path(args.out_root or PINNED_OUT_ROOT)
        digest = emit_pinned(out_root, calib, pin_id,
                             preload_dir=args.pinned_preload_dir,
                             library_tag=args.pinned_library_tag)
        print(f"[emit] pinned main arms {sorted(digest['per_teacher'])} "
              f"pin_id={pin_id} global_digest={digest['global_digest']} -> {out_root}",
              flush=True)
        return

    if not args.manifest:
        raise SystemExit("--manifest is required for the round-2 arms (the ws2c cell set)")
    teacher = args.teacher or "groot_tp"
    spec = TEACHERS[teacher]
    calib_entry = calibration_entry(calib, spec, teacher)
    manifest = json.loads(Path(args.manifest).read_text())
    control_cells = manifest["segments"]["ws2c"]["cells"]

    configs = weight_matrix()
    unknown = sorted(set(control_cells) - set(configs))
    if unknown:
        raise SystemExit(f"manifest ws2c cells not in the weight matrix: {unknown}")

    out_root = Path(args.out_root or DEFAULT_OUT_ROOT) / teacher
    main_index = emit_arm(out_root / MAIN_ARM, sorted(configs), configs, calib_entry,
                          teacher, text_ivf=True)
    ctrl_index = emit_arm(out_root / "control", list(control_cells), configs, calib_entry,
                          teacher, text_ivf=False)
    assert len(main_index) == 132, f"main arm must be 132 cells, got {len(main_index)}"
    assert len(ctrl_index) == 12, f"control arm must be 12 cells, got {len(ctrl_index)}"
    print(f"[emit] main={len(main_index)} control={len(ctrl_index)} -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
