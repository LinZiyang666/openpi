"""Non-manual ACT artifact-chain contract: REAL trainer & selector entry points
(lerobot CLI subprocess mocked, lerobot module stubbed) through to the sidecar
factory loading the SELECTED checkpoint from the prompt manifest."""

from __future__ import annotations

import json
import sys
import types

import h5py
import yaml


def _stub_lerobot(monkeypatch, loaded_paths):
    class _FakePolicy:
        @classmethod
        def from_pretrained(cls, path):
            loaded_paths.append(str(path))
            inst = cls()
            return inst

        def to(self, device):
            return self

        def eval(self):
            return self

    lerobot = types.ModuleType("lerobot")
    lerobot.__version__ = "0.3.3"
    act_mod = types.ModuleType("lerobot.policies.act.modeling_act")
    act_mod.ACTPolicy = _FakePolicy
    for name, mod in {
        "lerobot": lerobot,
        "lerobot.policies": types.ModuleType("lerobot.policies"),
        "lerobot.policies.act": types.ModuleType("lerobot.policies.act"),
        "lerobot.policies.act.modeling_act": act_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return loaded_paths


def test_train_select_sidecar_chain(tmp_path, monkeypatch):
    from exp.ablation_study import select_student_checkpoint as sel
    from exp.ablation_study import train_act

    # -- dataset layout the trainer consumes --
    ds_root = tmp_path / "per_task"
    task0 = ds_root / "task_0"
    (task0 / "dataset").mkdir(parents=True)
    (task0 / "prompt.txt").write_text("pick up the bowl\n")
    out_root = tmp_path / "act_out"

    # -- REAL trainer entry: lerobot CLI mocked to materialise 0.3.3 layout --
    def _fake_run(cmd, check, **kw):
        pm = out_root / "task_0" / "checkpoints" / "000100" / "pretrained_model"
        pm.mkdir(parents=True, exist_ok=True)
        (pm / "model.safetensors").write_bytes(b"w0")
        return types.SimpleNamespace(stdout="")

    monkeypatch.setattr(train_act.subprocess, "run",
                        lambda cmd, check=True, **kw: _fake_run(cmd, check, **kw))
    _stub_lerobot(monkeypatch, [])
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"steps": 1, "batch_size": 1}))
    manifest_path = tmp_path / "act_manifest.json"
    freeze_path = tmp_path / "freeze_act.yaml"
    monkeypatch.setattr(sys, "argv", [
        "train_act.py", "--recipe", str(recipe), "--dataset-root", str(ds_root),
        "--out", str(out_root), "--manifest-out", str(manifest_path),
        "--freeze-manifest-out", str(freeze_path),
    ])
    train_act.main()
    trained = json.loads(manifest_path.read_text())["pick up the bowl"]
    assert trained.endswith("checkpoints/000100/pretrained_model")

    # -- second candidate + val trajectories; REAL selector entry --
    pm2 = tmp_path / "cand2" / "checkpoints" / "000200" / "pretrained_model"
    pm2.mkdir(parents=True)
    (pm2 / "model.safetensors").write_bytes(b"w2")
    for cand, srs in ((trained, [True, False, False, False]),
                      (str(tmp_path / "cand2"), [True, True, False, False])):
        d = tmp_path / f"val_{abs(hash(cand)) % 100}"
        d.mkdir(exist_ok=True)
        for i, ok in enumerate(srs):
            with h5py.File(d / f"episode_{i}.h5", "w") as f:
                f.attrs["success"] = ok
        if cand == trained:
            val1 = d
        else:
            val2 = d
    monkeypatch.setattr(sys, "argv", [
        "select.py", "--candidates", f"{trained}:{val1}", f"{tmp_path / 'cand2'}:{val2}",
        "--anchor-sr", "0.9", "--band-low", "0.1", "--band-high", "0.9",
        "--freeze-manifest", str(freeze_path),
        "--update-act-manifest", str(manifest_path), "--prompt", "pick up the bowl",
    ])
    sel.main()
    frozen = yaml.safe_load(freeze_path.read_text())
    assert frozen["selection"]["selected_checkpoint"] == str(tmp_path / "cand2")
    assert frozen["selection"]["selected_sha256"]
    selected_in_manifest = json.loads(manifest_path.read_text())["pick up the bowl"]
    assert selected_in_manifest.endswith("checkpoints/000200/pretrained_model")

    # -- sidecar factory loads the SELECTED path from the updated manifest --
    loaded = []
    _stub_lerobot(monkeypatch, loaded)
    from exp.ablation_study.sidecar_server import make_act_policy

    make_act_policy(str(manifest_path), "cpu")
    assert loaded == [selected_in_manifest]
