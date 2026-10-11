import json

from exp.mw_finetune.groot.checkpoints import checkpoint_status, complete_checkpoint, preserve_incomplete


def test_interrupted_save_preserved_and_previous_snapshot_resumable(tmp_path):
    first = tmp_path / "checkpoint-10000"
    first.mkdir()
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json", "model.safetensors"]:
        (first / name).write_bytes(b"checkpoint")
    (first / "experiment_cfg").mkdir()
    (first / "experiment_cfg/metadata.json").write_text('{}')
    (first / "trainer_state.json").write_text(json.dumps({"global_step": 10000}))
    interrupted = tmp_path / "checkpoint-20000"
    interrupted.mkdir()
    (interrupted / "optimizer.pt").write_bytes(b"partial")
    complete, incomplete = checkpoint_status(tmp_path)
    assert complete == [first] and incomplete == [interrupted]
    renamed = preserve_incomplete(incomplete)
    assert len(renamed) == 1 and complete_checkpoint(first)
    assert not interrupted.exists()
    assert (tmp_path / renamed[0]).exists()
    assert checkpoint_status(tmp_path) == ([first], [])


def test_missing_weight_shard_is_incomplete(tmp_path):
    snapshot = tmp_path / "checkpoint-2"
    snapshot.mkdir()
    for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json"]:
        (snapshot / name).write_bytes(b"checkpoint")
    (snapshot / "trainer_state.json").write_text('{"global_step":2}')
    (snapshot / "experiment_cfg").mkdir()
    (snapshot / "experiment_cfg/metadata.json").write_text('{}')
    (snapshot / "model.safetensors.index.json").write_text('{"weight_map":{"a":"missing.safetensors"}}')
    assert not complete_checkpoint(snapshot)
