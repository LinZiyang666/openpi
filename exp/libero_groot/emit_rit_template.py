"""Rebase a ws_search winner recipe onto the W13 (snapshot-bearing) library.

The RIT line needs the search line's retrieval stack verbatim -- the weights
were chosen on that stack, so changing any of it invalidates the choice -- over
a library the search line could not use: only the W13 re-collection carries the
denoise snapshots a warm rung resumes from. Exactly three things therefore
differ from the source recipe, and this emitter is the only thing that may
change them:

*   ``backend.in_memory.preload_path`` -- the W13 tier-S3 library. Same 50
    trajectories (5 per task) the search ran on, re-collected with snapshots.
*   ``checkpoints.cp1.search_strategy.score_normalization`` -- Layer-1 params
    re-fitted on that library. The params are a property of the library's raw
    similarity distribution, not of the weights, so a swapped library needs
    them re-fitted even when the weights carry over.
*   ``denoise_schedule`` -- the GR00T loop the snapshots belong to. Without it
    the load guard cannot prove a warm rung's timesteps come from this head's
    schedule rather than Pi0.5's ten descending steps.

Everything else is copied byte-for-byte from the source, and the emitter
asserts that: a diff outside those three keys is a bug, not a customisation.

The gate and the judge are deliberately NOT set here. This is the base every
arm is built from; the arm emitters replace those two sections, and leaving
the source's ``always_search`` / ``always_hit`` in place means the unmodified
template is exactly the pure-cache baseline arm.

Usage:
  uv run python -m exp.libero_groot.emit_rit_template \
      --suite libero_spatial --out-dir exp/libero_groot/config/rit
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_10")
#: The W13 tier that holds 5 successful trajectories per task -- the same 50
#: the search and the gate line ran on, and the scale the Pi0.5 line's library
#: sits at (1018 / 2496 entries against our 1078 / 2598).
TIER = "S3"
SCHEDULE_ID = "groot_n15_k8_v1"
#: Server-side library root. weilandserver holds it on /data directly; h100
#: reaches the same tree through a /data -> /media/volume/OmniData symlink, so
#: one path string serves both lanes and no arm yaml is host-specific.
LIBRARY_ROOT = "/data/libero_cache/libraries_w13"
#: Keys this emitter is allowed to touch. Anything else differing from the
#: source recipe means the retrieval stack drifted and the weights no longer
#: describe what runs.
MUTABLE = (
    ("backend", "in_memory", "preload_path"),
    ("checkpoints", "cp1", "search_strategy", "score_normalization"),
    ("denoise_schedule",),
)


def library_path(suite: str) -> str:
    return f"{LIBRARY_ROOT}/{suite}/{suite}_w13_{TIER}.pkl"


def library_stem(suite: str) -> str:
    return f"{suite}_w13_{TIER}"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(doc: dict, path: tuple[str, ...]):
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set(doc: dict, path: tuple[str, ...], value) -> None:
    node = doc
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def normalization_section(calibration: dict, stem: str) -> dict:
    """The ``per_field`` block for one library, in the recipe's own shape.

    Reads the selected normalizer rather than the shortlist: the shortlist is a
    diagnostic, and the selection is what the calibrator committed to.
    """
    if stem not in calibration:
        raise SystemExit(
            f"calibration has no entry for {stem!r}; present: {sorted(calibration)}"
        )
    fields = {}
    for field, rec in calibration[stem]["fields"].items():
        selected = rec["selected"]
        fields[field] = {"method": selected["method"], "params": dict(selected["params"])}
    return {"type": "per_field", "fields": fields}


def build(source: dict, *, suite: str, calibration: dict) -> dict:
    doc = copy.deepcopy(source)
    _set(doc, ("backend", "in_memory", "preload_path"), library_path(suite))
    _set(
        doc,
        ("checkpoints", "cp1", "search_strategy", "score_normalization"),
        normalization_section(calibration, library_stem(suite)),
    )
    _set(doc, ("denoise_schedule",), SCHEDULE_ID)
    assert_only_mutable_changed(source, doc)
    return doc


def assert_only_mutable_changed(source: dict, built: dict) -> None:
    """Fail loudly if anything outside ``MUTABLE`` differs from the source."""
    a, b = copy.deepcopy(source), copy.deepcopy(built)
    for path in MUTABLE:
        for doc in (a, b):
            node = doc
            for key in path[:-1]:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                node.pop(path[-1], None)
    if a != b:
        raise SystemExit(
            "emitted recipe differs from the source outside the three mutable keys; "
            "the retrieval stack must stay verbatim"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument(
        "--calibration",
        default="exp/libero_groot/config/calibration_normalizers_w13_S3.json",
    )
    ap.add_argument("--out-dir", default="exp/libero_groot/config/rit")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir / args.suite
    src_path = out_dir / "source_template.yaml"
    if not src_path.exists():
        raise SystemExit(f"missing source recipe {src_path}")
    source = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    calibration = json.loads((REPO_ROOT / args.calibration).read_text(encoding="utf-8"))

    built = build(source, suite=args.suite, calibration=calibration)
    out_path = out_dir / "template.yaml"
    out_path.write_text(yaml.safe_dump(built, sort_keys=False), encoding="utf-8")

    record = {
        "protocol": "libero_groot_rit_template_v1",
        "suite": args.suite,
        "source_template": str(src_path.relative_to(REPO_ROOT)),
        "source_template_sha256": _sha(src_path),
        "calibration": args.calibration,
        "calibration_sha256": _sha(REPO_ROOT / args.calibration),
        "library_path": library_path(args.suite),
        "library_stem": library_stem(args.suite),
        "denoise_schedule": SCHEDULE_ID,
        "template_sha256": _sha(out_path),
    }
    (out_dir / "template_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}  sha256={record['template_sha256'][:16]}")
    print(f"  library {record['library_path']}")
    print(f"  schedule {SCHEDULE_ID}")


if __name__ == "__main__":
    main()
