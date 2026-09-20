"""CPU tests for exp/dp_nfe/mode_filter_datasets.py: label algorithms on synthetic trajectories, the held-out split, the
U/M selection rules and the native-format exporters (zarr / npy / hdf5 written and read back)."""

import json

import h5py
import numpy as np
import pytest

from exp.dp_nfe import mode_filter_datasets as MF


# ---------------------------------------------------------------- labels
# Independent transcription of the upstream 16-dim observation: block_pushing_multimodal.py::_compute_state returns an
# OrderedDict flattened in this order (sizes in parentheses). The test never uses the module's BP_* slices.
BP_FIELDS = [("block_translation", 2), ("block_orientation", 1), ("block2_translation", 2), ("block2_orientation", 1),
             ("effector_translation", 2), ("effector_target_translation", 2), ("target_translation", 2),
             ("target_orientation", 1), ("target2_translation", 2), ("target2_orientation", 1)]


def _bp_obs(T, **fields):
    """[T, 16] from per-field arrays ([T, size] or a constant vector), zeros elsewhere."""
    cols = []
    for name, size in BP_FIELDS:
        v = fields.get(name)
        if v is None:
            cols.append(np.zeros((T, size)))
        else:
            v = np.asarray(v, dtype=float)
            cols.append(np.broadcast_to(v, (T, size)) if v.ndim == 1 else v)
    obs = np.concatenate(cols, 1)
    assert obs.shape == (T, 16)
    return obs


def _bp_traj(first_block, assign, T=40):
    t0, t1 = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    b0 = np.tile([0.2, 0.2], (T, 1)); b1 = np.tile([-0.2, 0.2], (T, 1))
    t_first = 5
    mover, later = (b0, b1) if first_block == 0 else (b1, b0)
    for t in range(t_first, T):  # `first_block` moves first, the other one 10 steps later
        mover[t, 0] += 0.01 * (t - t_first + 1)
    for t in range(t_first + 10, T):
        later[t, 0] += 0.01 * (t - t_first - 9)
    tg = {0: t0, 1: t1}
    b0[-1] = tg[assign[0]]; b1[-1] = tg[assign[1]]
    # the effector target (dims 8:10) wanders the whole time: it must not be mistaken for a block target
    eff_t = np.stack([np.linspace(-2, 2, T), np.linspace(2, -2, T)], 1)
    return _bp_obs(T, block_translation=b0, block2_translation=b1, effector_target_translation=eff_t,
                   target_translation=t0, target2_translation=t1, effector_translation=np.array([0.0, -0.5]))


def test_blockpush_layout_constants_match_upstream():
    off = 0; spans = {}
    for name, size in BP_FIELDS:
        spans[name] = slice(off, off + size); off += size
    assert MF.BP_OBS_DIM == 16
    assert MF.BP_BLOCK == spans["block_translation"] and MF.BP_BLOCK2 == spans["block2_translation"]
    assert MF.BP_TARGET == spans["target_translation"] == slice(10, 12)
    assert MF.BP_TARGET2 == spans["target2_translation"] == slice(13, 15)
    assert MF.BP_EFFECTOR_TARGET == spans["effector_target_translation"] == slice(8, 10)


def test_blockpush_labels():
    assert MF.label_blockpush(_bp_traj(0, (0, 1))) == "first=0|assign=b0:t0,b1:t1"
    assert MF.label_blockpush(_bp_traj(1, (1, 0))) == "first=1|assign=b0:t1,b1:t0"
    assert MF.label_blockpush(_bp_traj(1, (0, 1))) == "first=1|assign=b0:t0,b1:t1"
    still = _bp_obs(10, target_translation=[1, 0], target2_translation=[-1, 0], block_translation=[0.2, 0.2], block2_translation=[-0.2, 0.2])
    assert MF.label_blockpush(still) == MF.UNKNOWN  # nothing moved
    both = _bp_traj(0, (0, 1)); both[5, 3:5] += 0.5  # simultaneous first move (block2 dims 3:5)
    assert MF.label_blockpush(both) == MF.UNKNOWN
    assert MF.label_blockpush(np.zeros((10, 13))) == MF.UNKNOWN  # wrong width is refused, not silently sliced


def test_kitchen_labels_order_and_set():
    T = 50
    obs = np.zeros((T, 60))
    obs[:, 22] = 0.0  # microwave open goal -0.75
    obs[20:, 22] = -0.75
    obs[:, 23:30] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    obs[10:, 23:30] = MF.KITCHEN_GOAL["kettle"]  # kettle done first at t=10
    lab = MF.label_kitchen(obs)
    assert lab == "set=kettle+microwave|order=kettle>microwave"
    mask = np.ones(T, dtype=bool); mask[15:] = False  # truncate before the microwave completes
    assert MF.label_kitchen(obs, mask) == "set=kettle|order=kettle"
    assert MF.label_kitchen(np.zeros((5, 60))) == MF.UNKNOWN


def test_kitchen_common_set_first_then_order_within_it():
    """Two-task completions are the most frequent *set* even though the single-task set is the most frequent *label*:
    U/M must be built inside the two-task set and the single-task episodes become unknown."""
    labs = {}
    for i in range(6):
        labs[f"a{i}"] = "set=kettle+microwave|order=kettle>microwave"
    for i in range(4):
        labs[f"b{i}"] = "set=kettle+microwave|order=microwave>kettle"
    for i in range(8):
        labs[f"c{i}"] = "set=kettle|order=kettle"   # most frequent label, but a different (smaller) set
    labs["d0"] = MF.UNKNOWN
    ids = sorted(labs)
    sel_labels, info = MF.kitchen_restrict_to_common_set(labs, ids)
    assert info["set"] == "kettle+microwave" and info["n_in_set"] == 10 and info["set_counts"] == {"kettle+microwave": 10, "kettle": 8}
    assert sel_labels["a0"] == "order=kettle>microwave" and sel_labels["b1"] == "order=microwave>kettle"
    assert all(sel_labels[f"c{i}"] == MF.UNKNOWN for i in range(8)) and sel_labels["d0"] == MF.UNKNOWN
    sel = MF.choose_subsets(sel_labels, ids, cap=10, subset_seed=1, min_n=2)
    assert sel["usable"] and sel["u_label"] == "order=kettle>microwave" and sel["n"] == 6
    a_ids = {f"a{i}" for i in range(6)}; b_ids = {f"b{i}" for i in range(4)}
    assert set(sel["U"]) == a_ids and set(sel["M"]) <= a_ids | b_ids and set(sel["M"]) & b_ids
    # without the restriction the generic rule would pick the single-task label as U (the counterexample)
    raw = MF.choose_subsets(labs, ids, cap=10, subset_seed=1, min_n=2)
    assert raw["u_label"] == "set=kettle|order=kettle"
    # no completed set at all -> everything unknown, no U/M
    empty, info2 = MF.kitchen_restrict_to_common_set({"x": MF.UNKNOWN}, ["x"])
    assert empty == {"x": MF.UNKNOWN} and info2["set"] is None


def test_pusht_side_label():
    T = 30
    st = np.zeros((T, 5)); st[:, 2:4] = [256, 256]; st[:, 4] = 0.0  # block heading +x
    st[:, 0] = np.linspace(0, 256, T); st[:, 1] = 300  # agent approaches from +y side -> left of heading (+x)
    assert MF.label_pusht(st) == "left"
    st[:, 1] = 212
    assert MF.label_pusht(st) == "right"
    far = st.copy(); far[:, 1] = 500
    assert MF.label_pusht(far, approach_radius=60) == MF.UNKNOWN


# ---------------------------------------------------------------- split & selection
def test_heldout_split_is_stable_and_disjoint():
    ids = [f"ep{i:03d}" for i in range(100)]
    tr, ho = MF.heldout_split(ids)
    tr2, ho2 = MF.heldout_split(list(reversed(ids)))
    assert tr == tr2 and ho == ho2 and len(ho) == 10 and not (set(tr) & set(ho)) and sorted(tr + ho) == ids


def test_choose_subsets_rules():
    rng = np.random.default_rng(0)
    ids = [f"ep{i:03d}" for i in range(120)]
    labels = {e: (["A"] * 60 + ["B"] * 40 + ["C"] * 10 + [MF.UNKNOWN] * 10)[i] for i, e in enumerate(ids)}
    sel = MF.choose_subsets(labels, ids, cap=50, subset_seed=1)
    assert sel["usable"] and sel["u_label"] == "A" and sel["n"] == 50
    assert len(sel["U"]) == 50 and all(labels[e] == "A" for e in sel["U"])
    assert len(sel["M"]) == 50 and len(set(sel["M"])) == 50
    m_counts = {l: sum(1 for e in sel["M"] if labels[e] == l) for l in "ABC"}
    assert sum(1 for l in m_counts if m_counts[l] >= 2) >= 2 and all(labels[e] != MF.UNKNOWN for e in sel["M"])
    assert sel["n_unknown"] == 10
    # determinism
    assert MF.choose_subsets(labels, ids, cap=50, subset_seed=1) == sel
    # too few episodes -> unusable, one label -> unusable
    small = {e: "A" if i < 15 else "B" for i, e in enumerate(ids[:30])}
    assert not MF.choose_subsets(small, ids[:30], cap=50, subset_seed=1)["usable"]
    assert not MF.choose_subsets({e: "A" for e in ids}, ids, cap=50, subset_seed=1)["usable"]


# ---------------------------------------------------------------- exporters
def test_zarr_export_roundtrip(tmp_path):
    zarr = pytest.importorskip("zarr")
    src = tmp_path / "src.zarr"
    g = zarr.open_group(str(src), mode="w"); d = g.create_group("data"); m = g.create_group("meta")
    ends = np.array([10, 25, 30]); st = np.arange(30 * 5, dtype=np.float32).reshape(30, 5)
    MF._zarr_put(d, "state", st); MF._zarr_put(d, "action", st[:, :2]); MF._zarr_put(m, "episode_ends", ends)
    dst = tmp_path / "sub.zarr"
    MF.write_zarr_subset(str(src), str(dst), [2, 0])
    r = zarr.open(str(dst), mode="r")
    assert list(np.asarray(r["meta/episode_ends"])) == [5, 15]
    assert np.allclose(np.asarray(r["data/state"])[:5], st[25:30]) and np.allclose(np.asarray(r["data/state"])[5:], st[:10])


def _write_mjl(path, qpos, nq=30, nv=29, nu=9, skip=40):
    """A MuJoCo .mjl log in the upstream binary layout: header iiiiiii + name, then records of
    1 + nq + nv + nu float32 (no mocap / sensor / user data); ``qpos`` [T, nq] is written once per kept record repeated
    ``skip`` times so that reading with skipamount=skip returns it."""
    import struct
    T = qpos.shape[0]
    recs = np.zeros((T * skip, 1 + nq + nv + nu), dtype="<f4")
    recs[:, 0] = np.arange(T * skip) * 0.01
    recs[:, 1:nq + 1] = np.repeat(qpos, skip, axis=0)
    name = b"synthetic"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("iiiiiii", nq, nv, nu, 0, 0, 0, len(name)) + name + recs.tobytes())


def test_kitchen_mjl_parse_label_and_export_roundtrip(tmp_path):
    src = tmp_path / "kitchen_demos_multitask"
    rng = np.random.default_rng(0)
    demos = {}
    for sess, name, done in (("friday_a", "d1", ("kettle", "microwave")), ("friday_a", "d2", ("microwave",)), ("postcorl_b", "d3", ())):
        q = np.zeros((50, 30)); q[:, 23:30] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; q[:, :9] = rng.normal(size=(50, 9)) * 0.01
        if "kettle" in done:
            q[10:, 23:30] = MF.KITCHEN_GOAL["kettle"]
        if "microwave" in done:
            q[20:, 22] = -0.75
        _write_mjl(src / sess / f"{name}.mjl", q); demos[f"{sess}/{name}.mjl"] = q
    ids = MF.list_kitchen_demos(str(src))
    assert ids == ["friday_a/d1.mjl", "friday_a/d2.mjl", "postcorl_b/d3.mjl"]
    q1 = MF.read_mjl_qpos(str(src / "friday_a/d1.mjl"))
    assert q1.shape == (50, 30) and np.allclose(q1, demos["friday_a/d1.mjl"], atol=1e-6)
    assert MF.label_kitchen(q1[:, :30]) == "set=kettle+microwave|order=kettle>microwave"
    assert MF.label_kitchen(MF.read_mjl_qpos(str(src / "friday_a/d2.mjl"))) == "set=microwave|order=microwave"
    assert MF.label_kitchen(MF.read_mjl_qpos(str(src / "postcorl_b/d3.mjl"))) == MF.UNKNOWN
    dst = tmp_path / "kitchen_U"
    MF.write_kitchen_subset(str(src), str(dst), ["postcorl_b/d3.mjl", "friday_a/d1.mjl"])
    assert MF.list_kitchen_demos(str(dst)) == ["friday_a/d1.mjl", "postcorl_b/d3.mjl"]
    assert (dst / "friday_a/d1.mjl").read_bytes() == (src / "friday_a/d1.mjl").read_bytes()
    # a truncated file is refused by the parser (build() records it as unknown)
    bad = tmp_path / "bad.mjl"; bad.write_bytes((src / "friday_a/d1.mjl").read_bytes()[:-7])
    with pytest.raises(ValueError):
        MF.read_mjl_qpos(str(bad))


def test_build_end_to_end_kitchen(tmp_path):
    src = tmp_path / "kitchen_demos_multitask"
    rng = np.random.default_rng(1)
    orders = [("kettle", "microwave")] * 30 + [("microwave", "kettle")] * 14 + [("kettle",)] * 20 + [()] * 6
    for i, order in enumerate(orders):
        q = np.zeros((40, 30)); q[:, 23:30] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; q[:, :9] = rng.normal(size=(40, 9)) * 0.01
        for j, task in enumerate(order):
            t0 = 5 + 10 * j
            if task == "kettle":
                q[t0:, 23:30] = MF.KITCHEN_GOAL["kettle"]
            else:
                q[t0:, 22] = -0.75
        _write_mjl(src / f"sess{i % 3}" / f"demo{i:03d}.mjl", q)
    man = MF.build("kitchen", str(src), str(tmp_path / "out"), cap=150, subset_seed=1)
    assert man["n_total"] == 70 and man["n_heldout"] == 7 and man["selection"]["usable"]
    assert man["kitchen"]["set"] == "kettle+microwave" and man["selection"]["u_label"] == "order=kettle>microwave"
    assert set(man["exports"]) == {"trainpool", "U", "M", "heldout"}
    for name in ("U", "M", "heldout", "trainpool"):
        d = tmp_path / "out" / f"kitchen_{name}"
        assert d.is_dir() and len(MF.list_kitchen_demos(str(d))) == len({"U": man["selection"]["U"], "M": man["selection"]["M"], "heldout": man["heldout"], "trainpool": man["train_pool"]}[name])
    assert all(v > 0 for v in man["lengths"].values())


def test_robomimic_export_and_operator_labels(tmp_path):
    src = tmp_path / "mh.hdf5"
    with h5py.File(src, "w") as f:
        data = f.create_group("data"); data.attrs["env_args"] = "{}"; data.attrs["total"] = 0
        for i in range(6):
            g = data.create_group(f"demo_{i}"); g.attrs["num_samples"] = 10 + i
            g.create_dataset("actions", data=np.zeros((10 + i, 7))); g.create_group("obs").create_dataset("x", data=np.zeros((10 + i, 3)))
        m = f.create_group("mask")
        m.create_dataset("better_operator_1", data=np.array([b"demo_0", b"demo_1", b"demo_2"]))
        m.create_dataset("better_operator_2", data=np.array([b"demo_3", b"demo_4"]))
    labels = MF.read_robomimic_operators(str(src))
    assert labels["demo_0"] == "better_operator_1" and labels["demo_4"] == "better_operator_2" and labels["demo_5"] == MF.UNKNOWN
    dst = tmp_path / "sub.hdf5"
    MF.write_robomimic_subset(str(src), str(dst), ["demo_4", "demo_1"])
    with h5py.File(dst, "r") as f:
        assert sorted(f["data"].keys()) == ["demo_0", "demo_1"]
        assert f["data/demo_0"].attrs["source_demo"] == "demo_4" and f["data"].attrs["total"] == 14 + 11
        assert f["data/demo_0/actions"].shape == (14, 7) and list(f["mask/train"][:]) == [b"demo_0", b"demo_1"]


def test_build_end_to_end_pusht(tmp_path):
    zarr = pytest.importorskip("zarr")
    src = tmp_path / "pusht.zarr"
    g = zarr.open_group(str(src), mode="w"); d = g.create_group("data"); m = g.create_group("meta")
    rng = np.random.default_rng(0); T = 30; n = 60
    states, ends, cur = [], [], 0
    for i in range(n):
        st = np.zeros((T, 5), dtype=np.float32); st[:, 2:4] = [256, 256]
        st[:, 0] = np.linspace(0, 256, T); st[:, 1] = 300 if i % 3 else 212  # 2/3 left, 1/3 right
        states.append(st); cur += T; ends.append(cur)
    MF._zarr_put(d, "state", np.concatenate(states)); MF._zarr_put(d, "action", np.zeros((cur, 2), dtype=np.float32))
    MF._zarr_put(m, "episode_ends", np.array(ends))
    man = MF.build("pusht", str(src), str(tmp_path / "out"), cap=80, subset_seed=1)
    sel = man["selection"]
    assert man["n_heldout"] == 6 and sel["usable"] and sel["u_label"] == "left" and sel["n"] >= MF.MIN_N
    assert not (set(sel["U"]) & set(man["heldout"])) and not (set(sel["M"]) & set(man["heldout"]))
    assert (tmp_path / "out" / "pusht_subset_manifest.json").exists() and set(man["exports"]) == {"trainpool", "U", "M", "heldout"}
    assert sorted(man["train_pool"]) == sorted(set(man["labels"]) - set(man["heldout"])) and len(man["train_pool"]) == 54
    r = zarr.open(str(tmp_path / "out" / "pusht_trainpool.zarr"), mode="r")
    assert len(np.asarray(r["meta/episode_ends"])) == 54
    j = json.loads((tmp_path / "out" / "pusht_subset_manifest.json").read_text())
    assert j["split_seed"] == MF.SPLIT_SEED and j["image_note"].startswith("skipped")
    # Supplying a path does not establish image availability.
    missing = MF.build("pusht", str(src), str(tmp_path / "noimg"), cap=80, subset_seed=1, image_src=str(src))
    assert not missing["image_exports"] and "data/img" in missing["image_note"]
    MF._zarr_put(d, "img", np.zeros((cur, 8, 8, 3), dtype=np.uint8))
    # With camera data present the image cells reuse the matched zarr subsets.
    man2 = MF.build("pusht", str(src), str(tmp_path / "out2"), cap=80, subset_seed=1, image_src=str(src))
    assert man2["image_exports"] == {n: man2["exports"][n] for n in ("U", "M", "heldout", "trainpool")}


def _mh_pair(tmp_path, n=80):
    """A lowdim mh hdf5 with operator masks and an image hdf5 with the same demo names and camera observations."""
    low = tmp_path / "low_dim_abs.hdf5"; img = tmp_path / "image_abs.hdf5"
    rng = np.random.default_rng(0)
    for path, with_img in ((low, False), (img, True)):
        with h5py.File(path, "w") as f:
            data = f.create_group("data"); data.attrs["env_args"] = "{}"; data.attrs["total"] = 0
            for i in range(n):
                g = data.create_group(f"demo_{i}"); L = 10 + i; g.attrs["num_samples"] = L
                g.create_dataset("actions", data=np.full((L, 7), i, dtype=np.float32))
                o = g.create_group("obs"); o.create_dataset("robot0_eef_pos", data=np.full((L, 3), i, dtype=np.float32))
                o.create_dataset("object", data=np.full((L, 14), i, dtype=np.float32))
                o.create_dataset("robot0_eef_quat", data=np.zeros((L, 4), dtype=np.float32))
                o.create_dataset("robot0_gripper_qpos", data=np.zeros((L, 2), dtype=np.float32))
                if with_img:
                    o.create_dataset("robot0_eye_in_hand_image", data=np.zeros((L, 84, 84, 3), dtype=np.uint8))
                    o.create_dataset("agentview_image", data=rng.integers(0, 255, size=(L, 84, 84, 3), dtype=np.uint8))
            m = f.create_group("mask")
            m.create_dataset("better_operator_1", data=np.array([f"demo_{i}".encode() for i in range(0, n, 2)]))
            m.create_dataset("better_operator_2", data=np.array([f"demo_{i}".encode() for i in range(1, n, 3)]))
    return low, img


def test_build_square_mh_exports_image_subsets_by_source_demo(tmp_path):
    low, img = _mh_pair(tmp_path)
    man = MF.build("square_mh", str(low), str(tmp_path / "out"), cap=50, subset_seed=1, image_src=str(img))
    assert man["selection"]["usable"] and set(man["image_exports"]) == {"U", "M", "heldout", "trainpool"}
    for name in ("U", "M", "heldout", "trainpool"):
        with h5py.File(tmp_path / "out" / f"square_mh_{name}.hdf5", "r") as fl, h5py.File(tmp_path / "out" / f"square_mh_image_{name}.hdf5", "r") as fi:
            demos = sorted(fl["data"].keys(), key=lambda s: int(s.split("_")[1]))
            assert demos == sorted(fi["data"].keys(), key=lambda s: int(s.split("_")[1]))
            for d in demos:
                src = fl[f"data/{d}"].attrs["source_demo"]
                assert fi[f"data/{d}"].attrs["source_demo"] == src            # same trajectory identity
                assert "agentview_image" in fi[f"data/{d}/obs"] and "agentview_image" not in fl[f"data/{d}/obs"]
                assert fi[f"data/{d}/obs/agentview_image"].shape[1:] == (84, 84, 3)
                assert np.array_equal(fi[f"data/{d}/actions"][:], fl[f"data/{d}/actions"][:])
            assert fi["data"].attrs["total"] == fl["data"].attrs["total"]
    assert man["image_exports"]["U"] != man["exports"]["U"]


# ---------------------------------------------------------------- cell manifest generator
def _subset_manifest(usable=True, image=False):
    m = {"selection": {"usable": usable, "n": 80, "u_label": "left", "reason": "" if usable else "one label"},
         "exports": {n: f"sha_{n}" for n in ("U", "M", "heldout", "trainpool")}, "kitchen": None}
    if image:
        m["image_exports"] = {n: f"isha_{n}" for n in ("U", "M", "heldout", "trainpool")}
    return m


def _normalizer_files(subsets, task, modality, sha="n" * 64):
    (subsets / f"{task}_{modality}_normalizer.pt").write_bytes(b"x")
    (subsets / f"{task}_{modality}_normalizer.pt.json").write_text(json.dumps({"sha256": sha,
        "source": {"dataset_sha256": "isha_trainpool" if modality == "image" else "sha_trainpool"}}))


def test_make_cells_binds_hashes_and_skips_unusable_or_unnormalized(tmp_path):
    import yaml
    from exp.dp_nfe import x0_cells
    tasks = x0_cells.resolve_raw_root(yaml.safe_load(open("exp/dp_nfe/config/x0_multimodal/tasks.yaml")), "/raw")
    tasks["frozen_budgets"] = {}  # this test exercises the command-line defaults
    subsets = tmp_path / "subsets"; subsets.mkdir()
    (subsets / "pusht_subset_manifest.json").write_text(json.dumps(_subset_manifest(image=True)))
    (subsets / "kitchen_subset_manifest.json").write_text(json.dumps(_subset_manifest(usable=False)))
    (subsets / "square_mh_subset_manifest.json").write_text(json.dumps(_subset_manifest(image=True)))
    _normalizer_files(subsets, "pusht", "lowdim"); _normalizer_files(subsets, "pusht", "image", "i" * 64)
    # square_mh has subsets but no frozen normaliser -> skipped with that reason
    m = x0_cells.make_cells(tasks, subsets, tmp_path / "cells", 20000, 10000, explore=False, image=True)
    assert len(m["by_arm"]["core"]) == 12 and len(m["by_arm"]["image"]) == 4 and m["by_arm"]["explore"] == []
    reasons = {(s["task"], s["arm"]): s["reason"] for s in m["skipped"]}
    assert reasons[("kitchen", "core")] == "one label" and "normalizer" in reasons[("square_mh", "core")]
    assert "normalizer" in reasons[("square_mh", "image")] and ("blockpush", "core") in reasons
    assert m["train_seeds"] == [42, 43, 44] and m["budgets"] == {"lowdim": 20000, "image": 10000}
    c = yaml.safe_load(open(tmp_path / "cells" / "pusht_lowdim_U_sample_s42_B20k.yaml"))
    assert c["head"] == "sample" and c["overrides"] == [f"task.dataset.zarr_path={subsets}/pusht_U.zarr"]
    assert c["heldout_path"].endswith("pusht_heldout.zarr") and c["heldout_path_key"] == "zarr_path"
    assert c["normalizer_path"].endswith("pusht_lowdim_normalizer.pt") and c["normalizer_sha256"] == "n" * 64
    ident = c["identity"]
    assert ident["subset_sha256"] == "sha_U" and ident["heldout_sha256"] == "sha_heldout" and ident["variant"] == "U"
    assert len(ident["subset_manifest_sha256"]) == 64 and ident["budget_id"] == "B20k"
    ci = yaml.safe_load(open(tmp_path / "cells" / "pusht_image_M_epsilon_s42_B10k.yaml"))
    assert ci["normalizer_sha256"] == "i" * 64 and ci["identity"]["subset_sha256"] == "isha_M" and ci["overrides"][0].endswith("pusht_M.zarr")
    m2 = x0_cells.make_cells(tasks, subsets, tmp_path / "cells2", 20000, 10000, explore=True, image=False)
    assert len(m2["by_arm"]["explore"]) == 2 * len(tasks["explore"]) and set(m2["cells"]) >= set(m["by_arm"]["core"])
    # image cells of square need the image exports: a lowdim-only manifest is skipped, never mirrored
    (subsets / "square_mh_subset_manifest.json").write_text(json.dumps(_subset_manifest(image=False)))
    _normalizer_files(subsets, "square_mh", "lowdim"); _normalizer_files(subsets, "square_mh", "image")
    m3 = x0_cells.make_cells(tasks, subsets, tmp_path / "cells3", 20000, 10000, explore=False, image=True)
    assert len(m3["by_arm"]["core"]) == 24 and not any(c.startswith("square_mh_image") for c in m3["cells"])
    assert any(s["task"] == "square_mh" and s["arm"] == "image" for s in m3["skipped"])


def test_per_task_budgets_and_wrong_normalizer_pool(tmp_path):
    import yaml
    from exp.dp_nfe import x0_cells
    tasks = x0_cells.resolve_raw_root(yaml.safe_load(open("exp/dp_nfe/config/x0_multimodal/tasks.yaml")), "/raw")
    tasks["frozen_budgets"] = {"lowdim": {"pusht": 50000, "square_mh": 100000}}
    subsets = tmp_path / "subsets"; subsets.mkdir()
    for task in ("pusht", "square_mh"):
        (subsets / f"{task}_subset_manifest.json").write_text(json.dumps(_subset_manifest()))
        _normalizer_files(subsets, task, "lowdim")
    out = tmp_path / "cells"
    manifest = x0_cells.make_cells(tasks, subsets, out, 20000, 10000, explore=True, image=False)
    assert manifest["budgets_by_task"] == tasks["frozen_budgets"]
    assert "pusht_lowdim_U_sample_s42_B50k" in manifest["cells"]
    square = yaml.safe_load((out / "square_mh_lowdim_M_sample_s44_B100k.yaml").read_text())
    assert square["budget_steps"] == 100000 and "task.dataset_type=mh" in square["overrides"]
    assert "can_ph_lowdim_full_epsilon_s42_B20k" in manifest["cells"]  # per-task override never leaks into defaults
    side = subsets / "pusht_lowdim_normalizer.pt.json"
    metadata = json.loads(side.read_text()); metadata["source"]["dataset_sha256"] = "other-pool"
    side.write_text(json.dumps(metadata))
    rejected = x0_cells.make_cells(tasks, subsets, tmp_path / "rejected", 20000, 10000, False, False)
    assert not any(c.startswith("pusht_") for c in rejected["cells"])
    assert any(s["task"] == "pusht" and "different pool" in s["reason"] for s in rejected["skipped"])


def test_image_source_requires_both_cameras_matched_observations_and_gripper(tmp_path):
    low, img = _mh_pair(tmp_path, n=2)
    assert MF.image_source_error("square_mh", str(low), str(img), ["demo_0", "demo_1"]) is None
    with h5py.File(img, "a") as f:  # a slightly different absolute position target (separate conversion) is tolerated
        f["data/demo_0/actions"][0, 0] += 0.03
    assert MF.image_source_error("square_mh", str(low), str(img), ["demo_0"]) is None
    with h5py.File(img, "a") as f:  # a different gripper command is not
        f["data/demo_0/actions"][0, -1] = -5
    assert "gripper" in MF.image_source_error("square_mh", str(low), str(img), ["demo_0"])
    with h5py.File(img, "a") as f:  # a different proprioceptive trajectory is another demonstration
        f["data/demo_0/actions"][0, -1] = 0
        f["data/demo_0/obs/robot0_eef_pos"][3, 1] += 1.0
    assert "observations differ" in MF.image_source_error("square_mh", str(low), str(img), ["demo_0"])
    with h5py.File(img, "a") as f:
        f["data/demo_0/obs/robot0_eef_pos"][3, 1] -= 1.0
        del f["data/demo_0/obs/robot0_eye_in_hand_image"]
    assert "robot0_eye_in_hand_image" in MF.image_source_error("square_mh", str(low), str(img), ["demo_0"])
