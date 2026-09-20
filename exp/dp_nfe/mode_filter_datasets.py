"""Label-filtered (U) vs label-mixed (M) subsets of the official Diffusion Policy datasets, with a common held-out split.

Trajectory labels (plan §3.1) are computed from the released data only:
    blockpush : which block moves first (>1e-3 from its initial xy, the env's own criterion) x final block->target
                assignment  (zarr ``data/obs`` 16-dim, DP's OrderedDict order)
    kitchen   : the set of completed sub-tasks and their completion order, re-derived from the qpos of each raw
                ``.mjl`` demo (parsed exactly like the upstream ``KitchenMjlLowdimDataset``: ``skipamount=40``; 30-dim
                qpos = 9 robot + 21 object joints, KitchenBase element indices); U/M are built inside the most frequent
                completed set only (``kitchen_restrict_to_common_set``)
    pusht     : lateral side (left/right in the T-block frame) on which the agent first enters an approach band of
                radius ``approach_radius`` px around the block centre (zarr ``data/state`` = [ax, ay, bx, by, theta])
    square_mh : operator id from the robomimic masks ``better_operator_1`` / ``better_operator_2``
Episodes whose label cannot be decided are ``unknown`` and excluded from both U and M.

Subset rules (plan §3.2): a 10% held-out set (split seed 20260918) is drawn first from every full dataset and never
enters U/M; U = the most frequent valid label (ties by label string); N = min(#U, task cap); M is a proportional draw
across the valid labels with at least two labels of >=2 episodes each; both draws are without replacement with a fixed
subset seed. Exports keep the native format so the official dataset classes load them unchanged:
zarr (pusht/blockpush), a directory of the selected raw ``<session>/<demo>.mjl`` files (kitchen, the ``kitchen_lowdim_abs``
recipe's ``KitchenMjlLowdimDataset`` input; the npy triple of ``kitchen_lowdim`` is unusable: 248/409 of its
``existence_mask`` rows are not prefix masks, so the upstream reader takes zero rows), hdf5 with renumbered ``data/demo_i``
(robomimic).

usage: python -m exp.dp_nfe.mode_filter_datasets --task pusht --src <zarr> --out <dir> [--cap 80] [--subset-seed 1]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SPLIT_SEED = 20260918
HELDOUT_FRAC = 0.10
CAPS = {"blockpush": 250, "kitchen": 150, "pusht": 80, "square_mh": 50}
MIN_N = 20
UNKNOWN = "unknown"

# --- blockpush 16-dim obs layout: the OrderedDict of block_pushing_multimodal.py::_compute_state flattened in order
#     block_translation(0:2) block_orientation(2) block2_translation(3:5) block2_orientation(5) effector_translation(6:8)
#     effector_target_translation(8:10) target_translation(10:12) target_orientation(12) target2_translation(13:15)
#     target2_orientation(15)
BP_OBS_DIM = 16
BP_BLOCK = slice(0, 2); BP_BLOCK2 = slice(3, 5); BP_EFFECTOR = slice(6, 8); BP_EFFECTOR_TARGET = slice(8, 10)
BP_TARGET = slice(10, 12); BP_TARGET2 = slice(13, 15)
BP_MOVE_EPS = 1e-3

# --- kitchen (diffusion_policy/env/kitchen/base.py)
KITCHEN_IDX = {"bottom burner": [11, 12], "top burner": [15, 16], "light switch": [17, 18], "slide cabinet": [19],
               "hinge cabinet": [20, 21], "microwave": [22], "kettle": [23, 24, 25, 26, 27, 28, 29]}
KITCHEN_GOAL = {"bottom burner": [-0.88, -0.01], "top burner": [-0.92, -0.01], "light switch": [-0.69, -0.05],
                "slide cabinet": [0.37], "hinge cabinet": [0.0, 1.45], "microwave": [-0.75],
                "kettle": [-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]}
KITCHEN_THRESH = 0.3


# ----------------------------------------------------------------------------- labels (pure numpy)
def label_blockpush(obs: np.ndarray) -> str:
    """obs [T, 16]. 'first=<0|1>|assign=<b0:t?,b1:t?>' or 'unknown'."""
    if obs.ndim != 2 or obs.shape[1] != BP_OBS_DIM or obs.shape[0] < 2:
        return UNKNOWN
    b = [obs[:, BP_BLOCK], obs[:, BP_BLOCK2]]
    first = None
    for t in range(1, obs.shape[0]):
        moved = [np.linalg.norm(b[i][t] - b[i][0]) > BP_MOVE_EPS for i in range(2)]
        if any(moved):
            if all(moved):
                return UNKNOWN  # simultaneous
            first = 0 if moved[0] else 1
            break
    if first is None:
        return UNKNOWN
    targets = [obs[-1, BP_TARGET], obs[-1, BP_TARGET2]]
    assign = []
    for i in range(2):
        d = [np.linalg.norm(b[i][-1] - targets[j]) for j in range(2)]
        if abs(d[0] - d[1]) < 1e-6:
            return UNKNOWN
        assign.append(int(np.argmin(d)))
    return f"first={first}|assign=b0:t{assign[0]},b1:t{assign[1]}"


def label_kitchen(obs: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """obs [T, >=30] (first 30 dims = qpos: 9 robot + 21 object joints). 'set=a+b+c+d|order=a>b>c>d' over the sub-tasks
    whose element distance to the KitchenBase goal first drops below the threshold, or 'unknown'. ``mask`` (optional
    row mask) selects the valid rows."""
    if obs.ndim != 2 or obs.shape[1] < 30:
        return UNKNOWN
    if mask is not None:
        obs = obs[mask.astype(bool)]
    if obs.shape[0] == 0:
        return UNKNOWN
    first: Dict[str, int] = {}
    for name, idx in KITCHEN_IDX.items():
        goal = np.asarray(KITCHEN_GOAL[name])
        dist = np.linalg.norm(obs[:, idx] - goal, axis=1)
        hits = np.nonzero(dist < KITCHEN_THRESH)[0]
        if len(hits):
            first[name] = int(hits[0])
    if not first:
        return UNKNOWN
    order = sorted(first, key=lambda k: (first[k], k))
    return "set=" + "+".join(sorted(first)) + "|order=" + ">".join(order)


def label_pusht(state: np.ndarray, approach_radius: float = 60.0) -> str:
    """state [T, 5] = [agent_x, agent_y, block_x, block_y, block_theta] (pixels). Side of first approach in the block
    frame: 'left' / 'right' / 'unknown' (never within the band, or exactly on the axis)."""
    if state.ndim != 2 or state.shape[1] < 5:
        return UNKNOWN
    d = state[:, :2] - state[:, 2:4]
    within = np.nonzero(np.linalg.norm(d, axis=1) < approach_radius)[0]
    if len(within) == 0:
        return UNKNOWN
    t = int(within[0])
    th = float(state[t, 4])
    heading = np.array([np.cos(th), np.sin(th)])
    cross = heading[0] * d[t, 1] - heading[1] * d[t, 0]
    if abs(cross) < 1e-9:
        return UNKNOWN
    return "left" if cross > 0 else "right"


def kitchen_restrict_to_common_set(labels: Dict[str, str], train_ids: Sequence[str]) -> Tuple[Dict[str, str], dict]:
    """Plan §3.2 for Kitchen: first fix the most frequent *completed-task set* on the training pool, then relabel the
    episodes of that set by their completion order ('order=...'); every other episode becomes ``unknown`` for U/M.
    Returns (order_labels, {'set': ..., 'set_counts': {...}, 'n_in_set': n})."""
    sets = collections.Counter()
    parsed = {}
    for e in train_ids:
        lab = labels.get(e, UNKNOWN)
        if lab == UNKNOWN or not lab.startswith("set="):
            continue
        st, order = lab.split("|", 1)
        parsed[e] = (st[len("set="):], order[len("order="):])
        sets[parsed[e][0]] += 1
    if not sets:
        return {e: UNKNOWN for e in train_ids}, {"set": None, "set_counts": {}, "n_in_set": 0}
    common = sorted(sets.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    out = {e: (f"order={parsed[e][1]}" if e in parsed and parsed[e][0] == common else UNKNOWN) for e in train_ids}
    return out, {"set": common, "set_counts": dict(sets), "n_in_set": sets[common]}


# ----------------------------------------------------------------------------- split + selection
def heldout_split(episode_ids: Sequence[str], frac: float = HELDOUT_FRAC, seed: int = SPLIT_SEED) -> Tuple[List[str], List[str]]:
    """Deterministic ``(train_pool_ids, heldout_ids)`` split: ``round(frac * n)`` episodes drawn by ``default_rng(seed)``."""
    ids = sorted(episode_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_hold = int(round(frac * len(ids)))
    hold = sorted(ids[i] for i in perm[:n_hold])
    train = sorted(ids[i] for i in perm[n_hold:])
    return train, hold


def choose_subsets(labels: Dict[str, str], train_ids: Sequence[str], cap: int, subset_seed: int,
                   min_n: int = MIN_N) -> dict:
    """Returns {'usable': bool, 'reason', 'U': [...], 'M': [...], 'u_label', 'counts', 'n'}."""
    valid = {e: labels[e] for e in train_ids if labels.get(e, UNKNOWN) != UNKNOWN}
    counts = collections.Counter(valid.values())
    res = {"usable": False, "reason": "", "U": [], "M": [], "u_label": None, "counts": dict(counts), "n": 0,
           "n_unknown": sum(1 for e in train_ids if labels.get(e, UNKNOWN) == UNKNOWN)}
    if len(counts) < 2:
        res["reason"] = "fewer than two valid labels"
        return res
    u_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    n = min(counts[u_label], cap)
    if n < min_n:
        res["reason"] = f"U has {n} < {min_n} episodes"
        return res
    rng = np.random.default_rng(subset_seed)
    pool_u = sorted(e for e, l in valid.items() if l == u_label)
    U = sorted(rng.choice(pool_u, size=n, replace=False).tolist())
    # proportional M: quota per label = round(n * share), fixed so that every label with >=2 pool episodes gets >=2
    total = sum(counts.values())
    quota = {l: int(round(n * c / total)) for l, c in counts.items()}
    for l in quota:
        quota[l] = max(2, quota[l]) if counts[l] >= 2 else 0
    # adjust to exactly n by trimming the largest quotas
    while sum(quota.values()) > n:
        l = max(quota, key=lambda k: (quota[k], k))
        if quota[l] <= 2:
            break
        quota[l] -= 1
    while sum(quota.values()) < n:
        l = max(counts, key=lambda k: (counts[k] - quota[k], k))
        if counts[l] - quota[l] <= 0:
            break
        quota[l] += 1
    if sum(quota.values()) != n or sum(1 for l in quota if quota[l] >= 2) < 2:
        res["reason"] = "cannot build a mixed subset with >=2 labels of >=2 episodes at N=%d" % n
        return res
    M: List[str] = []
    for l in sorted(quota):
        pool = sorted(e for e, v in valid.items() if v == l)
        if quota[l]:
            M += rng.choice(pool, size=quota[l], replace=False).tolist()
    res.update({"usable": True, "U": U, "M": sorted(M), "u_label": u_label, "n": n, "m_quota": quota,
                "overlap": sorted(set(U) & set(M))})
    return res


# ----------------------------------------------------------------------------- readers / writers
def _sha(path: pathlib.Path) -> str:
    from exp.dp_nfe.x0_identity import sha256_path
    return sha256_path(path)


def read_zarr_episodes(path: str, keys: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """``({key: data/<key> as array}, meta/episode_ends)`` of a DP replay-buffer zarr."""
    import zarr
    r = zarr.open(path, mode="r")
    data = {k: np.asarray(r["data"][k]) for k in keys}
    ends = np.asarray(r["meta"]["episode_ends"])
    return data, ends


def write_zarr_subset(src: str, dst: str, keep: Sequence[int]) -> None:
    """Copy the selected episodes (indices into meta/episode_ends order) into a new zarr with the same keys."""
    import zarr
    r = zarr.open(src, mode="r")
    ends = np.asarray(r["meta"]["episode_ends"]); starts = np.concatenate([[0], ends[:-1]])
    g = zarr.open_group(dst, mode="w")
    dg = g.create_group("data"); mg = g.create_group("meta")
    new_ends, cur = [], 0
    for i in keep:
        cur += int(ends[i] - starts[i]); new_ends.append(cur)
    for k in r["data"].keys():
        arr = r["data"][k]
        parts = [np.asarray(arr[int(starts[i]):int(ends[i])]) for i in keep]
        out = np.concatenate(parts, 0) if parts else np.zeros((0,) + arr.shape[1:], dtype=arr.dtype)
        _zarr_put(dg, k, out)
    _zarr_put(mg, "episode_ends", np.asarray(new_ends, dtype=np.int64))


def _zarr_put(group, name, arr):
    if hasattr(group, "create_array"):  # zarr 3
        group.create_array(name, data=arr)
    else:  # zarr 2
        group.create_dataset(name, data=arr)


def read_mjl_qpos(path: str, skipamount: int = 40) -> np.ndarray:
    """qpos [T, nq] of one MuJoCo ``.mjl`` log, transcribed from ``diffusion_policy.env.kitchen.kitchen_util.parse_mjl_logs``
    (header ``iiiiiii`` = nq, nv, nu, nmocap, nsensordata, nuserdata, name_len; records of 1 + nq + nv + nu + 7 nmocap +
    nsensordata + nuserdata float32; every ``skipamount``-th record kept, as the upstream dataset does)."""
    import struct
    raw = pathlib.Path(path).read_bytes()
    nq, nv, nu, nmocap, nsensordata, nuserdata, name_len = struct.unpack("iiiiiii", raw[:28])
    body = raw[28 + name_len:]
    dat = np.frombuffer(body, dtype="<f4")
    recsz = 1 + nq + nv + nu + 7 * nmocap + nsensordata + nuserdata
    if len(dat) % recsz != 0:
        raise ValueError(f"{path}: {len(dat)} floats not a multiple of the record size {recsz}")
    dat = dat.reshape(-1, recsz)
    return dat[::skipamount, 1:nq + 1].astype(np.float64)


def list_kitchen_demos(dataset_dir: str) -> List[str]:
    """Stable episode ids of a kitchen demo directory: sorted ``<session>/<file>.mjl`` relative paths (the upstream
    reader globs ``*/*.mjl``)."""
    d = pathlib.Path(dataset_dir)
    return sorted(p.relative_to(d).as_posix() for p in d.glob("*/*.mjl"))


def write_kitchen_subset(dataset_dir: str, dst: str, keep: Sequence[str]) -> None:
    """Copy the selected ``<session>/<file>.mjl`` demos (byte-identical) into ``dst`` keeping the session sub-directory,
    so ``KitchenMjlLowdimDataset(dataset_dir=dst)`` reads exactly these episodes."""
    d = pathlib.Path(dst)
    if d.exists():
        shutil.rmtree(d)
    for rel in keep:
        src = pathlib.Path(dataset_dir) / rel
        out = d / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)


def read_robomimic_operators(hdf5_path: str) -> Dict[str, str]:
    """``{demo_name: better_operator_1|better_operator_2|unknown}`` from the robomimic MH masks."""
    import h5py
    labels: Dict[str, str] = {}
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for name in ("better_operator_1", "better_operator_2"):
            if f"mask/{name}" not in f:
                continue
            for d in f[f"mask/{name}"][:]:
                labels[d.decode() if isinstance(d, bytes) else str(d)] = name
        for d in demos:
            labels.setdefault(d, UNKNOWN)
    return labels


def write_robomimic_subset(src: str, dst: str, keep: Sequence[str]) -> None:
    """New hdf5 with data/demo_0..demo_{n-1} (renumbered, original name kept in attrs['source_demo']) and mask/train."""
    import h5py
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        data = fo.create_group("data")
        for k, v in fi["data"].attrs.items():
            data.attrs[k] = v
        total = 0
        names = []
        for i, d in enumerate(keep):
            fi.copy(fi[f"data/{d}"], data, name=f"demo_{i}")
            data[f"demo_{i}"].attrs["source_demo"] = d
            total += int(fi[f"data/{d}"].attrs["num_samples"]); names.append(f"demo_{i}")
        data.attrs["total"] = total
        m = fo.create_group("mask")
        m.create_dataset("train", data=np.array(names, dtype="S"))


# ----------------------------------------------------------------------------- driver
def image_source_error(task: str, src: str, image_src: str, ids: Sequence[str]) -> Optional[str]:
    """Return a documented skip reason unless the source has matched trajectories and required cameras."""
    if not pathlib.Path(image_src).exists():
        return "image source missing"
    if task == "pusht":
        import zarr
        if pathlib.Path(src).resolve() != pathlib.Path(image_src).resolve():
            return "pusht image source must be the same replay zarr as the lowdim source"
        g = zarr.open(str(src), mode="r")
        if "img" not in g["data"] or g["data/img"].shape[0] != g["data/state"].shape[0]:
            return "pusht source has no aligned data/img"
    elif task == "square_mh":
        # Trajectory identity = same demo names, lengths and (bit-identical) proprioceptive / object observations. The
        # official image_abs.hdf5 and low_dim_abs.hdf5 were converted to absolute actions in two separate replay runs:
        # gripper columns agree, position targets differ by <= ~0.04 and rotations only by the axis-angle sign, so the
        # action arrays are *not* required to be equal; the max |pos| difference is recorded by build().
        import h5py
        with h5py.File(src, "r") as low, h5py.File(image_src, "r") as img:
            for eid in ids:
                if f"data/{eid}/actions" not in img:
                    return f"image source missing {eid}"
                a, b = low[f"data/{eid}/actions"], img[f"data/{eid}/actions"]
                if a.shape != b.shape:
                    return f"image actions shape differs from lowdim source for {eid}"
                if not np.array_equal(a[:, -1], b[:, -1]):
                    return f"image gripper actions differ from lowdim source for {eid}"
                for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"):
                    pa, pb = f"data/{eid}/obs/{key}", f"data/{eid}/obs/{key}"
                    if pa not in low or pb not in img or low[pa].shape != img[pb].shape or not np.allclose(low[pa][:], img[pb][:], atol=1e-6):
                        return f"image observations differ from lowdim source for {eid}/{key}"
                for key in ("agentview_image", "robot0_eye_in_hand_image"):
                    path = f"data/{eid}/obs/{key}"
                    if path not in img or img[path].shape[0] != a.shape[0]:
                        return f"image source missing/alignment error: {eid}/{key}"
    else:
        return f"no image variant for {task}"
    return None


def build(task: str, src: str, out: str, cap: Optional[int] = None, subset_seed: int = 1,
          approach_radius: float = 60.0, image_src: Optional[str] = None) -> dict:
    """Label, split and export one task: writes ``<out>/<task>_{trainpool,U,M,heldout}<ext>`` (+ ``_image_*`` for square
    when ``image_src`` is given), and ``<task>_subset_manifest.json`` with labels, counts, the selection, hashes and the
    kitchen set choice. Returns the manifest dict; ``selection.usable`` False means no U/M pair (reason recorded)."""
    cap = CAPS[task] if cap is None else cap
    outp = pathlib.Path(out); outp.mkdir(parents=True, exist_ok=True)
    if task in ("blockpush", "pusht"):
        key = "obs" if task == "blockpush" else "state"
        data, ends = read_zarr_episodes(src, [key])
        starts = np.concatenate([[0], ends[:-1]])
        ids = [f"ep{i:05d}" for i in range(len(ends))]
        labels = {}
        for i, e in enumerate(ids):
            seq = data[key][int(starts[i]):int(ends[i])]
            labels[e] = label_blockpush(seq) if task == "blockpush" else label_pusht(seq, approach_radius)
        lengths = {e: int(ends[i] - starts[i]) for i, e in enumerate(ids)}
        exporter = lambda keep, dst: write_zarr_subset(src, str(dst), [ids.index(e) for e in keep])
        ext = ".zarr"
    elif task == "kitchen":
        ids = list_kitchen_demos(src)
        labels, lengths = {}, {}
        for e in ids:
            try:
                q = read_mjl_qpos(str(pathlib.Path(src) / e))
            except Exception as exc:  # noqa: BLE001 - an unreadable demo is recorded, never guessed
                labels[e] = UNKNOWN; lengths[e] = 0
                continue
            labels[e] = label_kitchen(q[:, :30]); lengths[e] = int(q.shape[0])
        exporter = lambda keep, dst: write_kitchen_subset(src, str(dst), list(keep))
        ext = ""
    elif task == "square_mh":
        labels = read_robomimic_operators(src)
        ids = sorted(labels, key=lambda s: int(s.split("_")[1]))
        lengths = {}
        exporter = lambda keep, dst: write_robomimic_subset(src, str(dst), list(keep))
        ext = ".hdf5"
    else:
        raise ValueError(f"unknown task {task}")
    train_ids, hold_ids = heldout_split(ids)
    kitchen_info = None
    sel_labels = labels
    if task == "kitchen":
        sel_labels, kitchen_info = kitchen_restrict_to_common_set(labels, train_ids)
    sel = choose_subsets(sel_labels, train_ids, cap, subset_seed)
    manifest = {"task": task, "src": src, "src_sha256": _sha(pathlib.Path(src)), "split_seed": SPLIT_SEED,
                "heldout_frac": HELDOUT_FRAC, "subset_seed": subset_seed, "cap": cap, "approach_radius": approach_radius,
                "n_total": len(ids), "n_train_pool": len(train_ids), "n_heldout": len(hold_ids), "heldout": hold_ids,
                "train_pool": train_ids, "labels": labels, "selection_labels": sel_labels, "kitchen": kitchen_info,
                "lengths": lengths, "selection": sel, "exports": {}, "image_exports": {}}
    # the training pool (every non-held-out episode) is exported unconditionally: the shared normaliser is fitted on it
    exporter(train_ids, outp / f"{task}_trainpool{ext}")
    manifest["exports"]["trainpool"] = _sha(outp / f"{task}_trainpool{ext}")
    if sel["usable"]:
        for name, keep in (("U", sel["U"]), ("M", sel["M"]), ("heldout", hold_ids)):
            exporter(keep, outp / f"{task}_{name}{ext}")
            manifest["exports"][name] = _sha(outp / f"{task}_{name}{ext}")
        if image_src:
            why = image_source_error(task, src, image_src, ids)
            if why:
                manifest["image_note"] = f"skipped: {why}"
            elif task == "square_mh":  # same demo names in the image hdf5; exported by source id
                for name, keep in (("U", sel["U"]), ("M", sel["M"]), ("heldout", hold_ids), ("trainpool", train_ids)):
                    write_robomimic_subset(image_src, str(outp / f"{task}_image_{name}.hdf5"), list(keep))
                    manifest["image_exports"][name] = _sha(outp / f"{task}_image_{name}.hdf5")
                import h5py
                with h5py.File(src, "r") as low, h5py.File(image_src, "r") as img:
                    diffs = {e: float(np.abs(low[f"data/{e}/actions"][:, :3] - img[f"data/{e}/actions"][:, :3]).max()) for e in ids}
                manifest["image_action_note"] = ("separate absolute-action conversions of the same demonstrations: observations identical, "
                                                 "gripper equal, max |pos target| difference per demo recorded")
                manifest["image_action_pos_max_abs_diff"] = {"max": max(diffs.values()), "mean": float(np.mean(list(diffs.values())))}
            elif task == "pusht":  # the lowdim zarr already carries data/img: the same export serves the image policy
                manifest["image_exports"] = {n: manifest["exports"][n] for n in ("U", "M", "heldout", "trainpool")}
                manifest["image_note"] = "pusht zarr subsets carry data/img; image cells reuse them"
            else:
                manifest["image_note"] = f"no image variant for {task}"
        else:
            manifest["image_note"] = "skipped: no --image-src given"
    (outp / f"{task}_subset_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    """CLI: build the subsets of ``--task`` from ``--src`` into ``--out`` and print the selection summary."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(CAPS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--subset-seed", type=int, default=1)
    ap.add_argument("--approach-radius", type=float, default=60.0)
    ap.add_argument("--image-src", default=None, help="robomimic image hdf5 (square_mh) for the image confirmation cells")
    a = ap.parse_args()
    m = build(a.task, a.src, a.out, a.cap, a.subset_seed, a.approach_radius, a.image_src)
    s = m["selection"]
    print(f"{a.task}: usable={s['usable']} reason={s['reason']!r} counts={s['counts']} n={s['n']} u_label={s['u_label']}")


if __name__ == "__main__":
    main()
