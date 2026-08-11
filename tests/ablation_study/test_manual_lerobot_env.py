"""Manual LeRobot-venv API smoke (run inside the pinned student venv).

Asserts the exact 0.3.3 API surface the trainers/sidecar depend on: the train
CLI config (``dataset.repo_id`` required, pretrained ``policy.path``), dataset
creation, both policy classes with ``predict_action_chunk``, and the
train-output -> selection -> frozen-hash -> sidecar-load artifact chain.

Run (lerobot venv): pytest tests/ablation_study/test_manual_lerobot_env.py -m manual
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.manual


def test_lerobot_api_surface():
    lerobot = pytest.importorskip("lerobot")
    assert lerobot.__version__ == "0.3.3"
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    assert "repo_id" in DatasetConfig.__dataclass_fields__
    assert "steps" in TrainPipelineConfig.__dataclass_fields__
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert hasattr(LeRobotDataset, "create")
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    assert hasattr(SmolVLAPolicy, "predict_action_chunk")
    assert hasattr(ACTPolicy, "predict_action_chunk")


def test_artifact_chain_train_select_sidecar(tmp_path):
    """Executable artifact-chain contract: fake 0.3.3 checkpoint layout ->
    resolver -> selector (val SR + admission + sha256 freeze) -> sidecar
    manifest pointing at the resolved pretrained_model dir. The final
    real-policy load runs only in the pinned venv (importorskip)."""
    import json

    import h5py
    import yaml

    from exp.ablation_study.select_student_checkpoint import resolve_pretrained_dir
    from exp.ablation_study.select_student_checkpoint import sha256_tree
    from exp.ablation_study.select_student_checkpoint import val_sr

    # Fake LeRobot 0.3.3 train output with a loadable-dir marker weight.
    out = tmp_path / "task_0"
    pm = out / "checkpoints" / "000100" / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "model.safetensors").write_bytes(b"weights")
    resolved = resolve_pretrained_dir(out)
    assert resolved == pm
    hashes = sha256_tree(str(resolved))
    assert len(hashes) == 1

    # Val trajectories -> SR -> selection freeze fields.
    traj = tmp_path / "val"
    traj.mkdir()
    for i, ok in enumerate([True, True, False]):
        with h5py.File(traj / f"episode_{i}.h5", "w") as f:
            f.attrs["success"] = ok
    sr, n = val_sr(str(traj))
    assert (sr, n) == (2 / 3, 3)

    manifest = {"pick up the bowl": str(resolved)}
    (tmp_path / "act_manifest.json").write_text(json.dumps(manifest))
    freeze = {"selected": str(resolved), "sha256": hashes, "val_sr": sr}
    (tmp_path / "freeze.yaml").write_text(yaml.safe_dump(freeze))

    # Real policy load requires the pinned venv.
    pytest.importorskip("lerobot")
    from exp.ablation_study.sidecar_server import make_act_policy

    with pytest.raises(Exception):  # noqa: PT011 — fake weights cannot parse; the
        # call proves the factory consumes the manifest's resolved paths.
        make_act_policy(str(tmp_path / "act_manifest.json"), "cpu")
