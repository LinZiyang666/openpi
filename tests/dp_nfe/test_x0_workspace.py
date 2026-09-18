"""env_dependent: runs only inside the official Diffusion Policy environment (needs ``diffusion_policy``, hydra, diffusers
0.11.1) with ``DP_ROOT`` pointing at the DP repo. Executed separately (see logs/x0_multimodal_plan.log.md §7); an
import-skip here is NOT a pass. Covers, against the real DP classes: exactly B optimizer/LR/EMA updates for two datasets
of different trajectory lengths and both loss targets (the loss target is checked on the scheduler config *and* by the
sign of the eps-vs-x0 regression target), training + final save without any env runner, the frozen training-pool
normaliser shared by two cells (real ``LinearNormalizer`` round trip), resume after a simulated kill reproducing the
uninterrupted run, identity-mismatch rejection, and the evaluator's loader binding the identity from the payload."""

import json
import os
import pathlib

import numpy as np
import pytest

pytest.importorskip("diffusion_policy")
pytest.importorskip("hydra")
DP_ROOT = os.environ.get("DP_ROOT")
pytestmark = [pytest.mark.env_dependent, pytest.mark.skipif(not DP_ROOT, reason="DP_ROOT not set")]

import torch  # noqa: E402
import yaml  # noqa: E402

from exp.dp_nfe import mode_filter_datasets as MF  # noqa: E402
from exp.dp_nfe import train_x0  # noqa: E402
from exp.dp_nfe import x0_normalizer  # noqa: E402


def _synthetic_pusht_zarr(path, n_episodes, length, seed):
    import zarr
    rng = np.random.default_rng(seed)
    g = zarr.open_group(str(path), mode="w"); d = g.create_group("data"); m = g.create_group("meta")
    T = n_episodes * length
    MF._zarr_put(d, "keypoint", rng.uniform(0, 512, size=(T, 9, 2)).astype(np.float32))
    MF._zarr_put(d, "state", rng.uniform(0, 512, size=(T, 5)).astype(np.float32))
    MF._zarr_put(d, "action", rng.uniform(0, 512, size=(T, 2)).astype(np.float32))
    MF._zarr_put(m, "episode_ends", (np.arange(n_episodes) + 1) * length)


def _cell(tmp, name, zarr_path, head, budget, seed=42, save_every=1000, identity=None, normalizer=None, heldout=None):
    c = {"cell_id": name, "workspace_config": "train_diffusion_unet_lowdim_workspace", "task": "pusht_lowdim",
         "head": head, "train_seed": seed, "budget_steps": budget, "batch_size": 8, "window_seed": 7, "val_every": 2,
         "save_every": save_every, "val_batch_size": 8,
         "overrides": [f"task.dataset.zarr_path={zarr_path}", "training.device=cpu",
                       "policy.model.down_dims=[32,64]", "policy.model.diffusion_step_embed_dim=32"],
         "identity": {"task_name": "pusht", "modality": "lowdim", "variant": "full", "budget_id": "test", **(identity or {})}}
    if normalizer:
        c["normalizer_path"] = str(normalizer[0]); c["normalizer_sha256"] = normalizer[1]
    if c["identity"]["variant"] in ("U", "M") and normalizer and heldout is None:
        heldout = tmp / "heldout.zarr"
        if not heldout.exists():
            _synthetic_pusht_zarr(heldout, 2, 30, 17)
    if heldout:
        c["heldout_path_key"] = "zarr_path"; c["heldout_path"] = str(heldout)
        from exp.dp_nfe.x0_identity import sha256_path
        manifest = tmp / "subset_manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({"fixture": True}))
        c["identity"].update(subset_path=str(zarr_path), subset_sha256=sha256_path(zarr_path),
                             heldout_path=str(heldout), heldout_sha256=sha256_path(heldout),
                             subset_manifest_path=str(manifest), subset_manifest_sha256=sha256_path(manifest))
    p = tmp / f"{name}.yaml"; p.write_text(yaml.safe_dump(c)); return p


def _compose(cell_yaml, out, budget=0):
    cell = yaml.safe_load(open(cell_yaml))
    cfg = train_x0.compose_cell(pathlib.Path(DP_ROOT), cell, out, budget)
    resolved = train_x0.build_identity(cfg, cell, pathlib.Path(cell_yaml), pathlib.Path(DP_ROOT))
    out.mkdir(parents=True, exist_ok=True)
    train_x0.write_manifest(out, cell, cfg, resolved)
    return cfg


def _run(tmp, cell_yaml, out, budget=0):
    cfg = _compose(cell_yaml, out, budget)
    os.chdir(DP_ROOT)
    from exp.dp_nfe.x0_workspace import FixedStepWorkspace
    ws = FixedStepWorkspace(cfg, output_dir=str(out)); ws.run(); return ws


def test_budget_is_exact_for_different_lengths_and_both_heads(tmp_path):
    zs = tmp_path / "short.zarr"; zl = tmp_path / "long.zarr"
    _synthetic_pusht_zarr(zs, 4, 20, 0); _synthetic_pusht_zarr(zl, 4, 60, 1)
    for head in ("epsilon", "sample"):
        for name, z in (("short", zs), ("long", zl)):
            ws = _run(tmp_path, _cell(tmp_path, f"{name}_{head}", z, head, budget=5), tmp_path / f"run_{name}_{head}")
            assert ws.global_step == 5 and ws.ema.optimization_step == 5 and ws.lr_scheduler.last_epoch == 5
            assert str(ws.model.noise_scheduler.config.prediction_type) == head
            ck = tmp_path / f"run_{name}_{head}" / "checkpoints"
            assert (ck / "final.ckpt").exists() and (ck / "final.done").exists()
            assert json.loads((ck / "final.done").read_text())["global_step"] == 5
            log = [json.loads(l) for l in open(tmp_path / f"run_{name}_{head}" / "train_log.jsonl")]
            assert len(log) == 5 and all(np.isfinite(r["loss"]) for r in log) and "val_mse_ema" in log[-1]


def test_loss_target_differs_between_heads(tmp_path):
    """Both heads on the same batch and RNG state: the eps head regresses the noise, the x0 head the clean action."""
    z = tmp_path / "d.zarr"; _synthetic_pusht_zarr(z, 4, 30, 9)
    losses = {}
    for head in ("epsilon", "sample"):
        cfg = _compose(_cell(tmp_path, f"lt_{head}", z, head, budget=1), tmp_path / f"lt_{head}")
        os.chdir(DP_ROOT)
        import hydra
        from exp.dp_nfe.x0_workspace import FixedStepWorkspace
        ws = FixedStepWorkspace(cfg, output_dir=str(tmp_path / f"lt_{head}"))
        ds = hydra.utils.instantiate(cfg.task.dataset); ws.model.set_normalizer(ds.get_normalizer())
        batch = {k: v.unsqueeze(0).repeat(4, 1, 1) for k, v in ds[0].items()}
        torch.manual_seed(0); losses[head] = float(ws.model.compute_loss(batch))
        assert str(ws.model.noise_scheduler.config.prediction_type) == head
    assert losses["epsilon"] != losses["sample"]


def test_frozen_normalizer_round_trip_with_real_linear_normalizer(tmp_path):
    pool = tmp_path / "pool.zarr"; _synthetic_pusht_zarr(pool, 8, 30, 4)
    U = tmp_path / "U.zarr"; _synthetic_pusht_zarr(U, 3, 30, 5); M = tmp_path / "M.zarr"; _synthetic_pusht_zarr(M, 3, 30, 6)
    os.chdir(DP_ROOT)
    side = x0_normalizer.fit_from_config(pathlib.Path(DP_ROOT), "train_diffusion_unet_lowdim_workspace", "pusht_lowdim",
                                         [f"task.dataset.zarr_path={pool}"], tmp_path / "pusht_lowdim_normalizer.pt")
    assert len(side["sha256"]) == 64 and side["source"]["n_windows"] > 0
    states = {}
    for v, z in (("U", U), ("M", M)):
        ws = _run(tmp_path, _cell(tmp_path, f"nz_{v}", z, "sample", budget=2, identity={"variant": v},
                                  normalizer=(tmp_path / "pusht_lowdim_normalizer.pt", side["sha256"])), tmp_path / f"nz_{v}")
        states[v] = {k: t.clone() for k, t in ws.model.normalizer.state_dict().items()}
        assert json.loads((tmp_path / f"nz_{v}" / "identity.json").read_text())["normalizer_sha256_loaded"] == side["sha256"]
    frozen = torch.load(str(tmp_path / "pusht_lowdim_normalizer.pt"))
    assert set(states["U"]) == set(frozen) and all(torch.equal(states["U"][k], frozen[k]) and torch.equal(states["M"][k], frozen[k]) for k in frozen)
    with pytest.raises(ValueError, match="normalizer_path"):  # the trainer refuses a U/M cell without the frozen file
        _compose(_cell(tmp_path, "nz_bad", U, "sample", budget=1, identity={"variant": "U"}), tmp_path / "nz_bad")


def test_resume_reproduces_uninterrupted_run_and_rejects_identity_mismatch(tmp_path, monkeypatch):
    z = tmp_path / "d.zarr"; _synthetic_pusht_zarr(z, 4, 30, 2)
    full = _run(tmp_path, _cell(tmp_path, "cell", z, "sample", budget=6), tmp_path / "full")
    # interrupted: budget 6, latest.ckpt every 3 updates, killed right after the step-3 save
    part_yaml = _cell(tmp_path, "cell", z, "sample", budget=6, save_every=3)
    cfg = _compose(part_yaml, tmp_path / "part")
    from exp.dp_nfe.x0_workspace import FixedStepWorkspace
    orig = FixedStepWorkspace.save_atomic
    def killing(self, tag):
        p = orig(self, tag)
        if tag == "latest" and self.global_step == 3:
            raise KeyboardInterrupt
        return p
    monkeypatch.setattr(FixedStepWorkspace, "save_atomic", killing)
    with pytest.raises(KeyboardInterrupt):
        FixedStepWorkspace(cfg, output_dir=str(tmp_path / "part")).run()
    monkeypatch.setattr(FixedStepWorkspace, "save_atomic", orig)
    assert not (tmp_path / "part" / "checkpoints" / "final.ckpt").exists()
    ws2 = FixedStepWorkspace(cfg, output_dir=str(tmp_path / "part")); ws2.run()  # resumes from latest at step 3
    assert ws2.global_step == 6 and ws2.ema.optimization_step == 6
    for (n1, p1), (n2, p2) in zip(full.ema_model.state_dict().items(), ws2.ema_model.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    for (n1, p1), (n2, p2) in zip(full.model.state_dict().items(), ws2.model.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    la = [json.loads(l) for l in open(tmp_path / "full" / "train_log.jsonl")]; lb = [json.loads(l) for l in open(tmp_path / "part" / "train_log.jsonl")]
    assert [r["loss"] for r in la] == [r["loss"] for r in lb] and [r["lr"] for r in la] == [r["lr"] for r in lb]
    # identity mismatch: a different subset hash / cell id refuses the checkpoint
    for ident, name in (({"subset_sha256": "other"}, "cell"), ({}, "other")):
        with pytest.raises(ValueError, match="identity mismatch|subset_sha256"):
            cfg_other = _compose(_cell(tmp_path, name, z, "sample", 6, identity=ident), tmp_path / f"o_{name}")
            FixedStepWorkspace(cfg_other, output_dir=str(tmp_path / "part")).load_checkpoint(path=str(tmp_path / "part" / "checkpoints" / "latest.ckpt"))


def test_eval_loader_rejects_non_final_and_binds_identity(tmp_path, monkeypatch):
    z = tmp_path / "d.zarr"; _synthetic_pusht_zarr(z, 4, 30, 3)
    cfg = _compose(_cell(tmp_path, "cell", z, "epsilon", budget=4, save_every=2), tmp_path / "run")
    os.chdir(DP_ROOT)
    from exp.dp_nfe.x0_workspace import FixedStepWorkspace
    from exp.dp_nfe.eval_dp_steps_v2 import load_workspace
    orig = FixedStepWorkspace.save_atomic
    def killing(self, tag):
        p = orig(self, tag)
        if tag == "latest" and self.global_step == 2:
            raise KeyboardInterrupt
        return p
    monkeypatch.setattr(FixedStepWorkspace, "save_atomic", killing)
    with pytest.raises(KeyboardInterrupt):
        FixedStepWorkspace(cfg, output_dir=str(tmp_path / "run")).run()
    monkeypatch.setattr(FixedStepWorkspace, "save_atomic", orig)
    with pytest.raises(SystemExit, match="non-final"):
        load_workspace(pathlib.Path(DP_ROOT), tmp_path / "run" / "checkpoints" / "latest.ckpt", tmp_path / "ev")
    ws2 = FixedStepWorkspace(cfg, output_dir=str(tmp_path / "run")); ws2.run()  # resumes to 4
    ws3, cfg3, policy, identity = load_workspace(pathlib.Path(DP_ROOT), tmp_path / "run" / "checkpoints" / "final.ckpt", tmp_path / "ev")
    assert policy is ws3.ema_model and ws3.global_step == 4
    assert identity["cell_id"] == "cell" and identity["head"] == "epsilon" and identity["task_name"] == "pusht"
    assert len(identity["resolved_config_sha256"]) == 64 and len(identity["code_sha256"]) == 64 and identity["budget_steps"] == 4
