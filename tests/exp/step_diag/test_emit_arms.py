"""CPU tests for exp/step_diag/emit_arms.py: every emitted yaml keeps its source template's retrieval
identity and changes only the verdict layer, LIBERO cells come from the suite's own RIT template
(shadow + the self-start round's warm cells), warm start_t must be a snapshot timestep of the teacher's
schedule and one of the environment's warm_ts, GR00T cells stamp the denoise schedule, the index binds
every file by sha256, and the library check reads the payload contract instead of the file name."""

import hashlib
import json
import pathlib
import pickle
from types import SimpleNamespace

import pytest
import torch
import yaml

from exp.step_diag import emit_arms as EM
from exp.step_diag import envs as E
from openpi.cache.config import load_cache_config, validate_cache_config


@pytest.mark.parametrize("env_id", sorted(E.ENVS))
def test_cells_keep_retrieval_identity_and_change_only_the_verdict(env_id):
    env = E.ENVS[env_id]
    base, base_sha = EM.load_base(env_id)
    assert len(base_sha) == 64
    shadow = EM.build_shadow_cell(env_id, base)
    EM.verify_cell(env_id, shadow, base)
    cp1 = shadow["checkpoints"]["cp1"]
    assert cp1["gate"] == {"type": "always_search"} and shadow["write_policy"] == {"type": "never"}
    for key in ("keys", "key_builder", "backend"):
        assert shadow[key] == base[key]
    if env.policy == "pi05":
        assert cp1["judge"] == {"type": "threshold", "threshold": EM.UNREACHABLE_THRESHOLD}
    else:
        assert cp1["judge"] == {"type": "always_hit"} and shadow["denoise_schedule"] == env.schedule_id
    if env.benchmark == "robocasa365":
        for t in E.QB_WARM_TS[env.policy]:
            warm = EM.build_warm_cell(env_id, base, t)
            EM.verify_cell(env_id, warm, base)
            assert warm["checkpoints"]["cp1"]["judge"] == {"type": "always_warm_start", "start_t": t}
    else:
        for t in env.warm_ts:
            warm = EM.build_warm_cell(env_id, base, t)
            EM.verify_cell(env_id, warm, base)
            assert warm["checkpoints"]["cp1"]["judge"] == {"type": "always_warm_start", "start_t": t}
        if env.policy == "pi05":
            # the N4 template itself is a gated / tiered cell, not an admissible shadow cell
            with pytest.raises(ValueError):
                EM.verify_cell(env_id, base, base)
        else:
            # the GR00T LIBERO RIT template already IS the read-only shadow cell: identity
            assert shadow == base


def test_warm_start_t_must_be_a_snapshot_timestep():
    base, _ = EM.load_base("pi05_rc")
    with pytest.raises(ValueError):
        EM.build_warm_cell("pi05_rc", base, 0.25)
    gbase, _ = EM.load_base("groot_rc")
    with pytest.raises(ValueError):
        EM.build_warm_cell("groot_rc", gbase, 0.3)  # k=4 schedule has 0.75 / 0.5 / 0.25


def test_pi05_shadow_drops_warm_tiers_and_verify_rejects_them():
    base, _ = EM.load_base("pi05_libero_10")
    assert base["checkpoints"]["cp1"]["judge"].get("warm_tiers")  # the N4 template carries tiers
    shadow = EM.build_shadow_cell("pi05_libero_10", base)
    assert "warm_tiers" not in shadow["checkpoints"]["cp1"]["judge"]
    bad = yaml.safe_load(yaml.safe_dump(shadow))
    bad["checkpoints"]["cp1"]["judge"]["warm_tiers"] = [{"threshold": 0.9, "start_t": 0.5}]
    with pytest.raises(ValueError, match="warm tiers"):
        EM.verify_cell("pi05_libero_10", bad, base)


def test_arm_ids():
    assert EM.arm_id_of("warm", t=0.2) == "warm_t0.2" and EM.arm_id_of("warm", t=0.75) == "warm_t0.75"
    assert EM.arm_id_of("plain", k=2) == "plain_k2" and EM.arm_id_of("shadow") == "shadow"


def test_emit_writes_bound_index_and_valid_configs(tmp_path):
    index = EM.emit(tmp_path)
    assert set(index) == set(E.ENVS)
    for env_id, entry in index.items():
        env = E.ENVS[env_id]
        if env.benchmark == "robocasa365":
            assert set(entry["cells"]) == {"shadow", *(f"warm_t{t:g}" for t in E.QB_WARM_TS[env.policy])}
            assert set(entry["plain_arms"]) == {"full", *(f"plain_k{k}" for k in E.QB_PLAIN_KS[env.policy])}
            assert entry["plain_arms"]["full"]["exec_steps"] == env.k_full
            assert entry["qb"]["main_m"] == E.QB_MAIN_M[env.policy] and entry["qb"]["flat_episodes"] == 100
        else:
            assert set(entry["cells"]) == {"shadow", *(f"warm_t{t:g}" for t in env.warm_ts)} and "plain_arms" not in entry
            assert set(entry["libero_self"]["arms"]) == set(E.LIBERO_SELF_ARMS_BY_POLICY[env.policy])
        assert entry["source_sha256"] == hashlib.sha256((EM.REPO / entry["source_template"]).read_bytes()).hexdigest()
        for arm_id, cell in entry["cells"].items():
            path = tmp_path / cell["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == cell["sha256"]
            cfg = yaml.safe_load(path.read_text())
            assert cfg["backend"]["in_memory"]["preload_path"] == cell["library"]
            if arm_id.startswith("warm"):
                assert cell["remaining_steps"] == env.remaining_steps(cell["start_t"])
            else:
                assert cell["start_t"] is None and cell["remaining_steps"] is None
            validate_cache_config(load_cache_config(str(path)), check_files=False)
    on_disk = json.loads((tmp_path / "index.json").read_text())
    assert on_disk == index


def test_checked_in_arms_match_a_fresh_emit(tmp_path):
    """The committed yamls under config/arms are exactly what emit() produces from the templates."""
    fresh = EM.emit(tmp_path)
    checked = json.loads((pathlib.Path(EM.HERE) / "config" / "arms" / "index.json").read_text())
    for env_id in fresh:
        for arm_id, cell in fresh[env_id]["cells"].items():
            assert checked[env_id]["cells"][arm_id]["sha256"] == cell["sha256"], (env_id, arm_id)
        assert checked[env_id]["source_sha256"] == fresh[env_id]["source_sha256"]


def _fake_library(path, n=3, shape=(10, 32), ts=(0.1, 0.2, 0.3), steps=10, n_exec=7):
    entries = [SimpleNamespace(payload=SimpleNamespace(action_chunk=torch.zeros(*shape),
                                                       intermediates={t: torch.zeros(*shape) for t in ts},
                                                       denoising_num_steps=steps, schedule_id="groot_n15_k4_v1" if steps == 4 else "pi05_v1")) for _ in range(n)]
    mask = torch.zeros(shape[1], dtype=torch.bool)
    mask[:n_exec] = True
    with open(path, "wb") as f:
        pickle.dump({"entries": entries, "library_stats": SimpleNamespace(action_active_mask=mask)}, f)
    return path


def test_verify_library_reads_the_payload_contract(tmp_path):
    ok = _fake_library(tmp_path / "ok.pkl")
    info = EM.verify_library(ok, "pi05_libero_10")
    assert info["n_entries"] == 3 and info["action_shape"] == [10, 32] and info["n_executed"] == 7
    with pytest.raises(ValueError, match="snapshot"):
        EM.verify_library(_fake_library(tmp_path / "nosnap.pkl", ts=(0.1, 0.2)), "pi05_libero_10")
    with pytest.raises(ValueError, match="action_chunk"):
        EM.verify_library(_fake_library(tmp_path / "shape.pkl", shape=(50, 32)), "pi05_libero_10")
    assert EM.verify_library(_fake_library(tmp_path / "dims.pkl", n_exec=32), "pi05_libero_10")["n_executed"] == 7
    with pytest.raises(ValueError, match="denoising_num_steps"):
        EM.verify_library(_fake_library(tmp_path / "steps.pkl", steps=4), "pi05_libero_10")
    # RoboCasa GR00T: 16 x 32, 12 executed dims, snapshots 0.75 / 0.5
    g = _fake_library(tmp_path / "g.pkl", shape=(16, 32), ts=(0.75, 0.5, 0.25), steps=4, n_exec=12)
    assert EM.verify_library(g, "groot_rc")["snapshot_ts"] == [0.5, 0.75]
    index = EM.emit(tmp_path / "arms", ("groot_rc",), check_libraries=True, library_paths={"groot_rc": str(g)})
    assert index["groot_rc"]["library_check"]["n_entries"] == 3
    assert index["groot_rc"]["cells"]["shadow"]["library"] == str(g)


def test_identity_digests_are_cheap_and_cached(tmp_path):
    """checkpoint_digest never reads large weight files; library_digest hashes once into a sidecar."""
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "config.json").write_text("{}")
    big = ck / "model.safetensors"
    big.write_bytes(b"\0" * (E.SMALL_FILE_BYTES + 1))
    d1 = E.checkpoint_digest(ck)
    big.write_bytes(b"\1" * (E.SMALL_FILE_BYTES + 1))  # same size, different content: not read, same digest
    assert E.checkpoint_digest(ck) == d1
    big.write_bytes(b"\1" * (E.SMALL_FILE_BYTES + 2))  # size change is seen
    assert E.checkpoint_digest(ck) != d1
    (ck / "config.json").write_text('{"a": 1}')  # small files are content-hashed
    d3 = E.checkpoint_digest(ck)
    assert d3 != d1
    lib = tmp_path / "lib.pkl"
    lib.write_bytes(b"library-bytes")
    sha = E.library_digest(lib)
    side = tmp_path / "lib.pkl.sha256"
    assert side.is_file() and sha == hashlib.sha256(b"library-bytes").hexdigest()
    side.write_text(side.read_text().replace(sha, "f" * 64))  # cached value is trusted while size/mtime match
    assert E.library_digest(lib) == "f" * 64
    lib.write_bytes(b"library-bytes-2")
    assert E.library_digest(lib) == hashlib.sha256(b"library-bytes-2").hexdigest()


def test_config_sha_ignores_commit_and_runtime():
    base = dict(experiment_id="e", env=E.ENVS["pi05_rc"].to_json(), arm_id="full", mode="full", exec_steps=10,
                checkpoint="/c", checkpoint_sha256="x", cache_config=None, cache_config_sha256=None, library=None,
                library_sha256=None, code_commit="aaa", extras={"h_exec": 5, "runtime": {"host": "h100"}})
    a = E.RunManifest(**base).config_sha
    b = E.RunManifest(**dict(base, code_commit="bbb", extras={"h_exec": 5, "runtime": {"host": "weilandserver"}})).config_sha
    c = E.RunManifest(**dict(base, checkpoint_sha256="y")).config_sha
    d = E.RunManifest(**dict(base, extras={"h_exec": 6, "runtime": {"host": "h100"}})).config_sha
    assert a == b and a != c and a != d
