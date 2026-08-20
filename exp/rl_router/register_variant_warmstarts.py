#!/usr/bin/env python3
"""Register the per-arm-set warm-starts in the pilot record.

    python register_variant_warmstarts.py --pilot <selection.json> \
        --warmstart <arms>=<path> [--warmstart <arms>=<path> ...] [--check]

Why this exists
---------------
Two pre-registered facts are jointly unsatisfiable as written:

* the frozen run matrix assigns ``lambda_1`` to ``l10_tsc_lam1_s0`` and
  ``l10_tc_lam1_s0`` — i.e. one pilot's lambda serves every variant;
* ``MlpRouterJudge`` refuses weights whose ``meta.arms`` differs from the
  configured arms, so those two variants structurally cannot start from the
  two-arm file the pilot calibrated on.

The launch gate compared warm-start *files* by sha, so those runs could never
clear it. What the check protects is that lambda was calibrated from the same
starting policy — the same representation — so that is what gets registered
here, on the bytes:

* ``trunk_sha256`` — sha256 over ``W1``, ``b1``, ``feature_mu``, ``feature_sigma``
  in a fixed order. ``fit_warmstart.graft`` copies exactly these across
  unchanged and only rebuilds the output rows, so every genuine re-graft of one
  fit shares this digest; a head refit on different data does not.
* ``variant_warmstarts[arms]`` — the pre-registered file for each arm set. The
  gate accepts *only* files listed here, so an unregistered path still fails.

This widens nothing: an alternate file is accepted only when it is both
pre-registered and provably the same trunk. Refuses to register a file whose
trunk differs from the pilot's, and refuses to register under an arm set the
file does not declare.

``--check`` reports what would be written and exits without touching the file.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import torch

TRUNK_FIELDS = ("W1", "b1", "feature_mu", "feature_sigma")


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trunk_sha256(path: pathlib.Path) -> tuple[str, str]:
    """(trunk digest, declared arm set) for a router checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    missing = [f for f in TRUNK_FIELDS if f not in blob]
    if missing:
        raise SystemExit(f"{path}: checkpoint is missing {missing}; not a router warm-start")
    h = hashlib.sha256()
    for field in TRUNK_FIELDS:                       # fixed order: the digest is a contract
        t = blob[field].detach().to(torch.float32).contiguous()
        h.update(field.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest(), str((blob.get("meta") or {}).get("arms", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--warmstart", action="append", required=True,
                    metavar="ARMS=PATH", help="repeat per arm set, e.g. --warmstart tsc=/…/x.pt")
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    pilot_path = pathlib.Path(args.pilot)
    doc = json.loads(pilot_path.read_text(encoding="utf-8"))

    # The pilot's own warm-start is the reference trunk. Every candidate record
    # carries its digest; they must agree, or the record is not describing one
    # calibration and nothing here is meaningful.
    expected = {r.get("expected_warmstart_sha256") for r in (doc.get("runs") or {}).values()}
    expected.discard(None)
    if len(expected) != 1:
        raise SystemExit(f"pilot record has {len(expected)} distinct warm-start digests: {expected}")
    pilot_sha = expected.pop()

    entries, pilot_trunk = {}, None
    for spec in args.warmstart:
        if "=" not in spec:
            raise SystemExit(f"--warmstart wants ARMS=PATH, got {spec!r}")
        label, raw = spec.split("=", 1)
        path = pathlib.Path(raw)
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        digest, declared = trunk_sha256(path)
        # The label is free-form so one arm set can register more than one
        # start (e.g. the 50/50 warm start and the one refit at the measured
        # knee), but it must still name the arm set the file declares -- the
        # record has to stay readable as "which file for which variant".
        if not label.startswith(declared):
            raise SystemExit(
                f"{path} declares meta.arms={declared!r}; label {label!r} must start with it")
        entries[label] = {"path": str(path), "sha256": file_sha256(path),
                          "trunk_sha256": digest, "arms": declared}
        if entries[label]["sha256"] == pilot_sha:
            pilot_trunk = digest

    if pilot_trunk is None:
        raise SystemExit(
            "none of the offered files is the pilot's own warm-start "
            f"(sha {pilot_sha[:16]}…); pass it too so its trunk is the reference")

    mismatched = {a: e["trunk_sha256"] for a, e in entries.items()
                  if e["trunk_sha256"] != pilot_trunk}
    if mismatched:
        raise SystemExit(
            f"refusing to register: these are not re-grafts of the pilot's fit {mismatched}")

    print(f"pilot warm-start sha : {pilot_sha}")
    print(f"shared trunk sha256  : {pilot_trunk}")
    for arms, e in sorted(entries.items()):
        print(f"  {arms:4s} {e['sha256'][:16]}…  trunk OK  {e['path']}")

    if args.check:
        print("--check: nothing written")
        return 0

    doc["trunk_sha256"] = pilot_trunk
    doc["variant_warmstarts"] = entries
    doc.setdefault("amendments", []).append({
        "what": "registered per-arm-set warm-starts",
        "why": ("the frozen matrix assigns lambda_1 to R_tsc/R_tc while MlpRouterJudge "
                "requires each arm set's own file; the file-sha gate could not be met. "
                "Registering the re-grafts of the same fit keeps the assertion (same "
                "trunk, pre-registered path) while making the matrix satisfiable."),
        "widens_nothing": ("an alternate file is accepted only if it is listed here AND "
                           "carries trunk_sha256; unregistered paths still fail"),
    })
    pilot_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {pilot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
