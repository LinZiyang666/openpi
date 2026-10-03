"""Build library-specific row catalogs from read-only store arrays and frozen R7 stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

STORE = Path("/home/weiland/trace_runs/offline_search_store")
R7_FITS = Path("/home/weiland/trace_runs/os_closed_loop/r07_main/fits")
ROW_FIELDS = ("task_id", "episode", "step", "ep_len", "progress", "success", "prev", "next")
LIB_FIELDS = ("key_v0", "key_v1", "rs", "action") + ROW_FIELDS
STAGE_FIELDS = ("mode", "event_near", "rows_to_event", "stage_run")


def suite_key(suite):
    return {"libero_10": "l10", "libero_spatial": "spatial"}.get(suite, suite)


def catalog_key(model, suite, library):
    values = [str(model), str(suite_key(suite)), str(library)]
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", value) for value in values):
        raise ValueError("invalid model/suite/library catalog identity")
    return "_".join(values)


def parse_key(cell):
    normalized = cell.replace("_libero_10_", "_l10_", 1).replace("_libero_spatial_", "_spatial_", 1)
    parts = normalized.split("_", 2)
    if len(parts) != 3:
        raise ValueError("catalog cell must be <model>_<suite>_<lib>: " + cell)
    if catalog_key(*parts) != normalized:
        raise ValueError("invalid catalog cell: " + cell)
    return tuple(parts)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_path(directory, name):
    for filename in (name + ".npy", "rows." + name + ".npy"):
        path = directory / filename
        if path.exists():
            return path
    raise FileNotFoundError("missing store array %s in %s" % (name, directory))


def _embedded_stage(obj):
    from exp.offline_search.rounds.r07.stages.stages import StageTable
    pending, seen = [obj], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, StageTable):
            return value
        if isinstance(value, dict):
            pending.extend(value.get(k) for k in ("method", "stages", "stage_table", "follow_table") if k in value)
        else:
            pending.extend(getattr(value, k) for k in ("method", "base", "awm", "stages", "stage_table", "follow_table")
                           if getattr(value, k, None) is not None)
    return None


def _load_stage(path):
    """Read trusted local frozen fits only; never run fit() or a policy."""
    from exp.offline_search.rounds.r07.stages.stages import StageTable, SCHEMA
    with Path(path).open("rb") as f:
        value = pickle.load(f)
    if isinstance(value, dict) and value.get("schema") == SCHEMA:
        if hashlib.sha256(value["payload"]).hexdigest() != value["content_sha256"]:
            raise ValueError("stage artifact content mismatch: " + str(path))
        value = pickle.loads(value["payload"])
    table = _embedded_stage(value)
    if table is not None:
        table._freeze()
    return table


def _stage_candidates(model, suite, library, fits_root):
    size = 50 if library == "current" else 500
    # Standalone pi05 tables avoid deserializing a larger method when available.
    if model == "pi05":
        short = "sp" if suite == "spatial" else suite
        yield fits_root / ("stages_pi05_%s_%d.pkl" % (short, size))
    yield fits_root / ("r7_%s_%s_%d_SF1.pkl" % (model, suite, size))
    yield fits_root / ("r7_%s_%s_%d_CT%s.pkl" % (model, suite, size, "30" if size == 50 else "18"))


def build_catalog(run_root, cell, store_root=STORE, r7_fits=R7_FITS, stage_fit=None):
    model, suite, library = parse_key(cell)
    key = catalog_key(model, suite, library)
    directory = Path(store_root) / "library" / (model + "_" + suite) / library
    paths = {name: _array_path(directory, name) for name in LIB_FIELDS}
    arrays = {name: np.load(path, allow_pickle=False, mmap_mode="r") for name, path in paths.items()}
    n = len(arrays["task_id"])
    if any(value.ndim == 0 or len(value) != n for value in arrays.values()):
        raise ValueError("store arrays do not share library row axis: " + str(directory))
    for name in ROW_FIELDS:
        if arrays[name].shape != (n,):
            raise ValueError("catalog field must be one-dimensional: " + name)
    hashes = {name: _sha(path) for name, path in paths.items()}
    lib_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    frame = pd.DataFrame({name: np.asarray(arrays[name]) for name in ROW_FIELDS})
    frame.insert(0, "row", np.arange(n, dtype=np.int64))
    frame["lib_sha"] = lib_sha
    stage_info = dict(status="unavailable", reason="no frozen R7 StageTable found")
    candidates = [Path(stage_fit)] if stage_fit else _stage_candidates(model, suite, library, Path(r7_fits))
    for path in candidates:
        if not path.exists():
            if stage_fit:
                raise FileNotFoundError(str(path))
            continue
        table = _load_stage(path)
        if table is None:
            if stage_fit:
                raise ValueError("artifact contains no frozen StageTable: " + str(path))
            continue
        from exp.offline_search.rounds.r07.stages.stages import content_fingerprint
        if content_fingerprint(arrays, table.manifest) != table.fingerprint:
            raise ValueError("frozen StageTable belongs to a different library: " + str(path))
        for name in ("task_id", "episode", "step", "success"):
            if not np.array_equal(arrays[name], getattr(table, name)):
                raise ValueError("frozen stage row identity mismatch: " + name)
        for name in STAGE_FIELDS:
            value = np.asarray(getattr(table, name))
            if value.shape != (n,):
                raise ValueError("frozen stage field shape mismatch: " + name)
            frame[name] = value
        stage_info = dict(status="available", path=str(path), sha256=_sha(path), fingerprint=table.fingerprint,
                          retrieval_fingerprint=table.retrieval_fingerprint)
        break
    output = Path(run_root) / "catalog" / key
    output.mkdir(parents=True, exist_ok=True)
    target = output / "rows.parquet"
    part = output / "rows.parquet.part"
    frame.to_parquet(part, index=False)
    with part.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(str(part), str(target))
    metadata = dict(schema="osdebug.v1", model=model, suite=suite, lib=library, rows=n, lib_sha=lib_sha,
                    sha256=_sha(target), store_arrays={k: dict(path=str(paths[k]), sha256=hashes[k]) for k in hashes},
                    stages=stage_info)
    meta_part = output / "rows.json.part"
    with meta_part.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(meta_part), str(output / "rows.json"))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--store-root", type=Path, default=STORE)
    parser.add_argument("--r7-fits", type=Path, default=R7_FITS)
    parser.add_argument("--stage-fit", action="append", default=[], metavar="CELL=PATH")
    args = parser.parse_args()
    overrides = dict(value.split("=", 1) for value in args.stage_fit)
    for cell in args.cells:
        key = catalog_key(*parse_key(cell))
        result = build_catalog(args.run_root, cell, args.store_root, args.r7_fits, overrides.get(key, overrides.get(cell)))
        print(json.dumps(dict(catalog=key, rows=result["rows"], lib_sha=result["lib_sha"], stages=result["stages"])), flush=True)


if __name__ == "__main__":
    main()
