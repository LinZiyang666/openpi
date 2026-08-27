"""Materialise the dispatch-surface init pools from the official super-pool.

Authoritative provenance: the library's init map
(``exp/common/data/db/libero_cache/<suite>_init_map.json``) records, per
library episode, which official init index (``orig_init_state_idx``) produced
it — 5 per task for libero_spatial. This script:

  1. Runs the D_lib census against that record (NOT against low-dim
     robot_state similarity). Any provenance anomaly — missing map, wrong
     episode count, incomplete mapping, per-task usage != 5 — aborts loudly.
  2. Deducts the D_lib indices from the official 50/task super-pool and splits
     the remaining 45 with a fixed seed into query C (15: 5 fit + 10 cal) and
     test A' (30), per task independently.
  3. Materialises A' and C as ``materialize_apool``-format directories
     (torch-saved state arrays, one ``<task>.init`` per task), asserts the
     three-way disjointness explicitly, and writes a split manifest with
     per-pool digests.

Usage:
  uv run python -m exp.dispatch_surface.split_init_pools \
      --suite libero_spatial \
      --init-map exp/common/data/db/libero_cache/libero_spatial_init_map.json \
      --apool-dir exp/common/data/db_init/libero/libero_spatial_apool \
      --out-root exp/dispatch_surface/data/init_pools \
      --seed 20260827
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random

import numpy as np
import torch

EXPECTED_TASKS = 10
OFFICIAL_PER_TASK = 50
DLIB_PER_TASK = 5
FIT_PER_TASK = 5
CAL_PER_TASK = 10
QUERY_PER_TASK = FIT_PER_TASK + CAL_PER_TASK  # 15
TEST_PER_TASK = OFFICIAL_PER_TASK - DLIB_PER_TASK - QUERY_PER_TASK  # 30


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _state_array(value) -> np.ndarray:
    """Canonical CPU array for exact init-state identity comparisons."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _normalise_attr(value):
    """Turn h5py/numpy attr values into deterministic JSON-compatible values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify_dlib_content(
    rows: list[dict], h5_dir: pathlib.Path, official_pool_dir: pathlib.Path,
) -> dict:
    """Content-level D_lib provenance verification (G2R3-B1).

    For every init-map row: resolve ``h5_path`` canonically (the resolved
    file must exist AND live inside ``h5_dir`` — never a basename join),
    open it as real HDF5 and compare the row's recorded ``attrs`` snapshot
    against the file's live attrs; resolve both the library collection init
    pool and the independently supplied official pool, then require the
    collection's ``subset_init_state_idx`` state to be byte-identical to the
    official ``orig_init_state_idx`` state.  Merely proving that both indices
    are in range is insufficient: swapping two official identities would
    silently corrupt the D_lib/query/test partition.
    Basename collisions inside ``h5_dir`` are refused outright. Returns the
    three-way content digests (H5 files / official pools / init map) for the
    split manifest so downstream records can compare against them.
    """
    h5_root = h5_dir.resolve()
    official_root = official_pool_dir.resolve()
    if not h5_root.is_dir():
        raise SystemExit(f"library H5 directory does not exist: {h5_dir}")
    if not official_root.is_dir():
        raise SystemExit(f"official pool directory does not exist: {official_pool_dir}")
    all_h5 = list(h5_root.rglob("*.h5"))
    by_name: dict[str, list[pathlib.Path]] = {}
    for p in all_h5:
        by_name.setdefault(p.name, []).append(p)
    collisions = {n: v for n, v in by_name.items() if len(v) > 1}
    if collisions:
        raise SystemExit(f"basename collisions inside {h5_dir}: {sorted(collisions)[:3]}")

    h5_digests: dict[str, str] = {}
    pool_digests: dict[str, str] = {}
    collection_pool_digests: dict[str, str] = {}
    mapped_h5: set[pathlib.Path] = set()
    for row in rows:
        required = {
            "h5_path", "trajectory_id", "task_name", "prompt", "entry_count", "attrs",
            "init_path", "subset_init_state_idx", "full_init_path", "orig_init_state_idx",
        }
        missing = sorted(required - set(row))
        if missing:
            raise SystemExit(f"init map row missing content-authority fields {missing}")
        raw = str(row.get("h5_path", ""))
        resolved = pathlib.Path(raw).resolve()
        if not resolved.is_file():
            raise SystemExit(f"init map h5_path does not exist: {raw}")
        if not resolved.is_relative_to(h5_root):
            raise SystemExit(f"init map h5_path {raw} escapes the library dir {h5_dir}")
        if resolved in mapped_h5:
            raise SystemExit(f"init map maps the same H5 more than once: {resolved}")
        mapped_h5.add(resolved)
        try:
            import h5py

            with h5py.File(resolved, "r") as h5:
                live_attrs = {k: _normalise_attr(h5.attrs[k]) for k in h5.attrs}
                step_count = sum(k.startswith("step_") for k in h5.keys())
        except Exception as exc:  # noqa: BLE001 — any unreadable file is a refusal
            raise SystemExit(f"{raw} is not readable HDF5: {exc}") from exc
        recorded = {k: _normalise_attr(v) for k, v in (row.get("attrs") or {}).items()}
        if recorded != live_attrs:
            raise SystemExit(f"{raw}: live HDF5 attrs differ from the init-map snapshot")
        if resolved.stem != str(row["trajectory_id"]):
            raise SystemExit(
                f"{raw}: filename trajectory {resolved.stem!r} != map "
                f"trajectory_id {row['trajectory_id']!r}"
            )
        if live_attrs.get("task") != row["prompt"]:
            raise SystemExit(f"{raw}: HDF5 task attr does not match the map prompt")
        if step_count != int(row["entry_count"]):
            raise SystemExit(
                f"{raw}: contains {step_count} step groups but map records "
                f"entry_count={row['entry_count']}"
            )
        h5_digests[str(resolved.relative_to(h5_root))] = _file_sha256(resolved)

        init_path = pathlib.Path(str(row["full_init_path"])).resolve()
        if not init_path.is_file() or not init_path.is_relative_to(official_root):
            raise SystemExit(
                f"init map full_init_path is not a file inside the official pool dir: {init_path}"
            )
        expected_name = f"{row['task_name']}.init"
        if init_path.name != expected_name:
            raise SystemExit(
                f"{raw}: official pool filename {init_path.name!r} != {expected_name!r}"
            )
        states = torch.load(init_path, weights_only=False)
        if len(states) != OFFICIAL_PER_TASK:
            raise SystemExit(
                f"{init_path}: official pool has {len(states)} states, "
                f"expected {OFFICIAL_PER_TASK}"
            )
        orig = int(row["orig_init_state_idx"])
        if not 0 <= orig < len(states):
            raise SystemExit(
                f"{raw}: orig_init_state_idx {orig} out of range for pool "
                f"{init_path} ({len(states)} states)"
            )
        pool_digests[init_path.name] = _file_sha256(init_path)

        collection_path = pathlib.Path(str(row["init_path"])).resolve()
        if not collection_path.is_file() or collection_path.name != expected_name:
            raise SystemExit(f"{raw}: invalid library collection init pool {collection_path}")
        collection_states = torch.load(collection_path, weights_only=False)
        if len(collection_states) != DLIB_PER_TASK:
            raise SystemExit(
                f"{collection_path}: collection pool has {len(collection_states)} states, "
                f"expected {DLIB_PER_TASK}"
            )
        subset = int(row["subset_init_state_idx"])
        if not 0 <= subset < len(collection_states):
            raise SystemExit(
                f"{raw}: subset_init_state_idx {subset} out of range for "
                f"{collection_path} ({len(collection_states)} states)"
            )
        if not np.array_equal(_state_array(collection_states[subset]), _state_array(states[orig])):
            raise SystemExit(
                f"{raw}: collection init[{subset}] does not equal official "
                f"init[{orig}] — orig_init_state_idx identity is false"
            )
        collection_pool_digests[collection_path.name] = _file_sha256(collection_path)
    actual_h5 = {p.resolve() for p in all_h5}
    if mapped_h5 != actual_h5:
        missing = sorted(str(p) for p in actual_h5 - mapped_h5)
        extra = sorted(str(p) for p in mapped_h5 - actual_h5)
        raise SystemExit(
            "init map H5 set is not exactly the rebuild source tree: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        "h5": h5_digests,
        "official_pools": pool_digests,
        "collection_init_pools": collection_pool_digests,
    }


def census_dlib_inits(
    init_map_path: pathlib.Path, *, h5_dir: pathlib.Path, official_pool_dir: pathlib.Path,
) -> tuple[dict[int, dict], dict]:
    """Return ({task_id: {...}}, content_digests) or abort.

    ``h5_dir`` is REQUIRED: the map's claims are verified against the actual
    library files at content level, never accepted as fields (G2R3-B1).
    """
    if not init_map_path.is_file():
        raise SystemExit(f"init map missing: {init_map_path} — provenance unavailable, aborting")
    rows = json.loads(init_map_path.read_text())
    if len(rows) != EXPECTED_TASKS * DLIB_PER_TASK:
        raise SystemExit(
            f"init map has {len(rows)} rows, expected {EXPECTED_TASKS * DLIB_PER_TASK}"
        )
    per_task: dict[int, dict] = {}
    grouped = collections.defaultdict(list)
    for row in rows:
        for key in (
            "task_id", "task_name", "orig_init_state_idx", "full_init_path",
            "subset_init_state_idx", "init_path",
        ):
            if row.get(key) is None:
                raise SystemExit(f"init map row missing '{key}': {row.get('h5_path')}")
        grouped[int(row["task_id"])].append(row)
    if sorted(grouped) != list(range(EXPECTED_TASKS)):
        raise SystemExit(f"init map task ids {sorted(grouped)} != 0..{EXPECTED_TASKS - 1}")
    for task_id, task_rows in grouped.items():
        indices = sorted(int(r["orig_init_state_idx"]) for r in task_rows)
        if len(set(indices)) != DLIB_PER_TASK:
            raise SystemExit(
                f"task {task_id}: D_lib occupies {sorted(set(indices))} "
                f"({len(set(indices))} distinct official inits, expected {DLIB_PER_TASK})"
            )
        names = {r["task_name"] for r in task_rows}
        if len(names) != 1:
            raise SystemExit(f"task {task_id}: inconsistent task_name in init map: {names}")
        subset_indices = sorted(int(r["subset_init_state_idx"]) for r in task_rows)
        if subset_indices != list(range(DLIB_PER_TASK)):
            raise SystemExit(
                f"task {task_id}: collection subset indices {subset_indices} "
                f"!= 0..{DLIB_PER_TASK - 1}"
            )
        init_paths = {pathlib.Path(str(r["init_path"])).resolve() for r in task_rows}
        if len(init_paths) != 1:
            raise SystemExit(f"task {task_id}: rows reference multiple collection init pools")
        per_task[task_id] = {"task_name": names.pop(), "indices": indices}
    content_digests = verify_dlib_content(
        rows, pathlib.Path(h5_dir), pathlib.Path(official_pool_dir),
    )
    content_digests["init_map"] = _file_sha256(init_map_path)
    return per_task, content_digests


def split_remaining(dlib_indices: list[int], seed: int, task_id: int) -> dict[str, list[int]]:
    """Deterministically split the 45 remaining official indices per task."""
    remaining = sorted(set(range(OFFICIAL_PER_TASK)) - set(dlib_indices))
    if len(remaining) != OFFICIAL_PER_TASK - DLIB_PER_TASK:
        raise SystemExit(f"task {task_id}: expected 45 remaining inits, got {len(remaining)}")
    rng = random.Random(seed * 1000 + task_id)
    rng.shuffle(remaining)
    return {
        "fit": sorted(remaining[:FIT_PER_TASK]),
        "cal": sorted(remaining[FIT_PER_TASK:QUERY_PER_TASK]),
        "test": sorted(remaining[QUERY_PER_TASK:]),
    }


def _sha256_states(states) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest()


def materialize_pool(
    apool_dir: pathlib.Path,
    out_dir: pathlib.Path,
    assignment: dict[int, dict],
    pool_indices_key_list: list[str],
) -> dict[str, dict]:
    """Write per-task .init files with the selected official states subset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, dict] = {}
    for task_id in sorted(assignment):
        task_name = assignment[task_id]["task_name"]
        src = apool_dir / f"{task_name}.init"
        if not src.is_file():
            raise SystemExit(f"official pool file missing: {src}")
        states = torch.load(src, weights_only=False)
        if len(states) != OFFICIAL_PER_TASK:
            raise SystemExit(f"{src}: has {len(states)} inits, expected {OFFICIAL_PER_TASK}")
        indices: list[int] = []
        for key in pool_indices_key_list:
            indices.extend(assignment[task_id][key])
        subset = states[sorted(indices)]
        dest = out_dir / f"{task_name}.init"
        torch.save(subset, dest)
        digests[task_name] = {
            "indices": sorted(indices),
            "count": len(indices),
            "sha256": _sha256_states(subset),
        }
    return digests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--init-map", required=True)
    ap.add_argument("--apool-dir", required=True,
                    help="materialised official pool (50/task), e.g. db_init/libero/<suite>_apool")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--library-h5-dir", required=True,
                    help="library-source H5 dir; every init-map row is verified "
                         "against it at content level (mandatory, G2R3-B1)")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    dlib, content_digests = census_dlib_inits(
        pathlib.Path(args.init_map),
        h5_dir=pathlib.Path(args.library_h5_dir),
        official_pool_dir=pathlib.Path(args.apool_dir),
    )
    print(f"D_lib census OK: {EXPECTED_TASKS} tasks x {DLIB_PER_TASK} official inits each")

    assignment: dict[int, dict] = {}
    for task_id, info in dlib.items():
        splits = split_remaining(info["indices"], args.seed, task_id)
        # Three-way disjointness is by construction; assert it anyway.
        pools = [set(info["indices"]), set(splits["fit"]), set(splits["cal"]), set(splits["test"])]
        for i in range(len(pools)):
            for j in range(i + 1, len(pools)):
                if pools[i] & pools[j]:
                    raise SystemExit(f"task {task_id}: pools overlap: {pools[i] & pools[j]}")
        assert len(splits["test"]) == TEST_PER_TASK
        assignment[task_id] = {"task_name": info["task_name"], "dlib": info["indices"], **splits}

    out_root = pathlib.Path(args.out_root)
    apool_dir = pathlib.Path(args.apool_dir)
    test_digests = materialize_pool(apool_dir, out_root / "test_aprime", assignment, ["test"])
    query_digests = materialize_pool(apool_dir, out_root / "query_c", assignment, ["fit", "cal"])

    manifest = {
        "suite": args.suite,
        "seed": args.seed,
        "init_map": str(args.init_map),
        "apool_dir": str(apool_dir),
        "library_h5_dir": str(pathlib.Path(args.library_h5_dir)),
        "quota": {"dlib": DLIB_PER_TASK, "fit": FIT_PER_TASK, "cal": CAL_PER_TASK,
                  "test": TEST_PER_TASK},
        "assignment": {
            str(tid): {k: v for k, v in info.items()} for tid, info in assignment.items()
        },
        "pool_digests": {"test_aprime": test_digests, "query_c": query_digests},
        "dlib_content_digests": content_digests,
    }
    manifest_path = out_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"pools materialised under {out_root}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
