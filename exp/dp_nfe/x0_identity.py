"""Identity helpers shared by the trainer, the workspace, the evaluator and the aggregator (no DP dependency).

A *cell identity* is the flat dict written to ``<run>/manifest.json`` / ``identity.json`` and pickled into every
checkpoint: the cell fields (task, modality, variant, head, train_seed, budget_id, ...) plus the provenance hashes
(subset / normaliser / resolved config / code / dependencies). Two identities are the same iff every field of the union
of their keys compares equal as a string.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any, Dict, List

# fields that define a cell for the aggregator (the matrix key); the remaining identity fields are provenance
CELL_KEY_FIELDS = ("task_name", "modality", "variant", "head", "train_seed", "budget_id")


def identity_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Field-by-field comparison over the union of keys (string-compared); returns ``["key: a != b", ...]``, empty when
    the two identities are identical."""
    out = []
    for k in sorted(set(a) | set(b)):
        if str(a.get(k, "<missing>")) != str(b.get(k, "<missing>")):
            out.append(f"{k}: {a.get(k, '<missing>')!r} != {b.get(k, '<missing>')!r}")
    return out


def sha256_file(path: pathlib.Path) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: pathlib.Path) -> str:
    """Hash a file or native dataset directory using the subset export convention."""
    path = pathlib.Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"dataset missing: {path}")
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"dataset empty: {path}")
    for p in files:
        h.update(str(p.relative_to(path)).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def expected_identity(cell: dict) -> dict:
    """Fields frozen by a cell YAML, excluding provenance added at launch."""
    ident = dict(cell.get("identity", {}))
    ident.update({k: cell[k] for k in ("cell_id", "head", "train_seed", "budget_steps")})
    ident.update(batch_size=int(cell.get("batch_size", 256)),
                 window_seed=int(cell.get("window_seed", 1000 + int(cell["train_seed"]))),
                 warmup_steps=min(500, int(cell["budget_steps"]) // 10))
    if cell.get("normalizer_sha256"):
        ident["normalizer_sha256"] = cell["normalizer_sha256"]
    if cell.get("_yaml_sha256"):
        ident["cell_yaml_sha256"] = cell["_yaml_sha256"]
    return ident


def frozen_identity_diff(actual: dict, cell: dict) -> List[str]:
    """Compare every frozen field, allowing additional launch provenance."""
    expected = expected_identity(cell)
    return identity_diff(expected, {k: actual[k] for k in expected if k in actual})


def verify_data_identity(identity: dict) -> None:
    """Reject changed dataset bytes before training or diagnostic sampling."""
    for name in ("subset", "heldout", "subset_manifest"):
        path, expected = identity.get(name + "_path"), identity.get(name + "_sha256")
        if path and expected and sha256_path(pathlib.Path(path)) != expected:
            raise ValueError(f"{name}_sha256 differs from frozen data: {path}")


def cell_key(identity: Dict[str, Any]) -> str:
    """``<task_name>_<modality>_<variant>_<head>_s<train_seed>_<budget_id>`` -- the cell_id convention of x0_cells.py."""
    return (f"{identity['task_name']}_{identity['modality']}_{identity['variant']}_{identity['head']}"
            f"_s{identity['train_seed']}_{identity['budget_id']}")
