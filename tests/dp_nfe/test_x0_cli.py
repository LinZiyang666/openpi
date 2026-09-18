"""env_dependent (DP env + ``DP_ROOT``): the actual command-line entry points in **fresh processes** (G2 R1-B1) and the
sampler against the **real diffusers 0.11.1 schedulers** (G2 R1-N1).

* ``python -m exp.dp_nfe.train_x0 --dry-run`` for a core-like lowdim cell (held-out + frozen normaliser), an image cell
  (hybrid workspace, ``${eval:...}`` in the official configs) and an explore cell (no held-out) -- each must compose,
  resolve and write ``manifest.json`` with the provenance hashes without importing the workspace first; a second launch
  keeps the manifest byte-for-byte; a launch with a different budget under the same directory is refused;
* ``python -m exp.dp_nfe.x0_normalizer`` freezes a normaliser from a synthetic training pool;
* ``python -m exp.dp_nfe.x0_cells`` + ``x0_queue status`` on the generated matrix;
* ``dp_sampler.sample_ddim`` (eps head, k=100 where the leading and trailing grids coincide) equals a chain of
  ``diffusers.DDIMScheduler.step`` calls, and ``ddpm_step`` equals ``DDPMScheduler.step`` (fixed_small) with the same
  noise, on the real scheduler objects of the official config."""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("diffusion_policy")
pytest.importorskip("hydra")
diffusers = pytest.importorskip("diffusers")
DP_ROOT = os.environ.get("DP_ROOT")
pytestmark = [pytest.mark.env_dependent, pytest.mark.skipif(not DP_ROOT, reason="DP_ROOT not set")]

import torch  # noqa: E402
import yaml  # noqa: E402

from exp.dp_nfe import dp_sampler as S  # noqa: E402
from tests.dp_nfe.test_x0_workspace import _synthetic_pusht_zarr  # noqa: E402

OPENPI = pathlib.Path(__file__).resolve().parents[2]


def _fresh(args, cwd=None):
    env = {**os.environ, "PYTHONPATH": f"{OPENPI}:{DP_ROOT}:" + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-m", *args], cwd=cwd or DP_ROOT, env=env, capture_output=True, text=True)


def _cell_yaml(path, cid, workspace, task, overrides, head="epsilon", budget=5, heldout=None, normalizer=None, identity=None):
    c = {"cell_id": cid, "workspace_config": workspace, "task": task, "head": head, "train_seed": 42, "budget_steps": budget,
         "batch_size": 4, "window_seed": 1, "val_every": 2, "save_every": 2, "val_batch_size": 4, "overrides": overrides,
         "identity": {"task_name": task.split("_")[0], "modality": "image" if "image" in task else "lowdim", "variant": "full",
                      "budget_id": "Bt", **(identity or {})}}
    if heldout:
        c["heldout_path_key"] = "zarr_path"; c["heldout_path"] = str(heldout)
    from exp.dp_nfe.x0_identity import sha256_path
    dataset_path = pathlib.Path(next(o.split("=", 1)[1] for o in overrides if o.startswith("task.dataset.zarr_path=")))
    c["identity"].update(subset_path=str(dataset_path), subset_sha256=sha256_path(dataset_path))
    if heldout:
        manifest = path.with_suffix(".subset.json")
        manifest.write_text(json.dumps({"test_fixture": True}))
        c["identity"].update(heldout_path=str(heldout), heldout_sha256=sha256_path(heldout),
                             subset_manifest_path=str(manifest), subset_manifest_sha256=sha256_path(manifest))
    if normalizer:
        c["normalizer_path"], c["normalizer_sha256"] = str(normalizer[0]), normalizer[1]
    path.write_text(yaml.safe_dump(c)); return path


def test_fresh_process_dry_run_core_image_explore_and_manifest_guard(tmp_path):
    pool = tmp_path / "pool.zarr"; _synthetic_pusht_zarr(pool, 6, 20, 0)
    U = tmp_path / "U.zarr"; _synthetic_pusht_zarr(U, 3, 20, 1); ho = tmp_path / "ho.zarr"; _synthetic_pusht_zarr(ho, 2, 20, 2)
    nz = tmp_path / "pusht_lowdim_normalizer.pt"
    r = _fresh(["exp.dp_nfe.x0_normalizer", "--dp-root", DP_ROOT, "--workspace-config", "train_diffusion_unet_lowdim_workspace",
                "--task", "pusht_lowdim", "--override", f"task.dataset.zarr_path={pool}", "--out", str(nz)])
    assert r.returncode == 0 and "NORMALIZER FROZEN" in r.stdout, r.stderr[-2000:]
    sha = json.loads(pathlib.Path(str(nz) + ".json").read_text())["sha256"]
    cells = {
        "core": _cell_yaml(tmp_path / "core.yaml", "pusht_lowdim_U_epsilon_s42_Bt", "train_diffusion_unet_lowdim_workspace", "pusht_lowdim",
                           [f"task.dataset.zarr_path={U}", "training.device=cpu"], heldout=ho, normalizer=(nz, sha), identity={"variant": "U"}),
        "image": _cell_yaml(tmp_path / "image.yaml", "pusht_image_U_sample_s42_Bt", "train_diffusion_unet_hybrid_workspace", "pusht_image",
                            [f"task.dataset.zarr_path={U}", "training.device=cpu"], head="sample", heldout=ho, normalizer=(nz, sha), identity={"variant": "U"}),
        "explore": _cell_yaml(tmp_path / "explore.yaml", "pusht_lowdim_full_sample_s42_Bt", "train_diffusion_unet_lowdim_workspace", "pusht_lowdim",
                              [f"task.dataset.zarr_path={pool}", "training.device=cpu"], head="sample"),
    }
    for name, y in cells.items():
        out = tmp_path / f"out_{name}"
        r = _fresh(["exp.dp_nfe.train_x0", "--dp-root", DP_ROOT, "--cell", str(y), "--out", str(out), "--dry-run"])
        assert r.returncode == 0 and "DRY RUN ok" in r.stdout, (name, r.stderr[-3000:])
        man = json.loads((out / "manifest.json").read_text()); ident = man["identity"]
        assert len(ident["resolved_config_sha256"]) == 64 and len(ident["code_sha256"]) == 64 and ident["deps"]
        resolved = yaml.safe_load((out / "resolved_config.yaml").read_text())
        assert resolved["_target_"].endswith("FixedStepWorkspace") and resolved["x0"]["budget_steps"] == 5
        assert resolved["task"]["dataset"]["pad_before"] == resolved["n_obs_steps"] - 1  # ${eval:...} resolved
        assert resolved["task"]["dataset"]["max_train_episodes"] is None
        if name != "explore":
            assert resolved["x0"]["heldout_dataset"]["zarr_path"] == str(ho) and resolved["x0"]["normalizer_sha256"] == sha
        raw = (out / "manifest.json").read_bytes()
        r2 = _fresh(["exp.dp_nfe.train_x0", "--dp-root", DP_ROOT, "--cell", str(y), "--out", str(out), "--dry-run"])
        assert r2.returncode == 0 and (out / "manifest.json").read_bytes() == raw
    # a different budget under an existing cell directory is refused before anything is written
    r3 = _fresh(["exp.dp_nfe.train_x0", "--dp-root", DP_ROOT, "--cell", str(cells["core"]), "--out", str(tmp_path / "out_core"), "--dry-run", "--budget", "7"])
    assert r3.returncode != 0 and "refusing to overwrite" in (r3.stderr + r3.stdout)


def test_fresh_process_short_training_then_eval_loader(tmp_path):
    """A 3-update lowdim cell end to end in a fresh process, then the evaluator's loader in another."""
    U = tmp_path / "U.zarr"; _synthetic_pusht_zarr(U, 3, 20, 1)
    y = _cell_yaml(tmp_path / "c.yaml", "pusht_lowdim_full_epsilon_s42_Bt", "train_diffusion_unet_lowdim_workspace", "pusht_lowdim",
                   [f"task.dataset.zarr_path={U}", "training.device=cpu", "policy.model.down_dims=[32,64]",
                    "policy.model.diffusion_step_embed_dim=32"], budget=3)
    out = tmp_path / "run"
    r = _fresh(["exp.dp_nfe.train_x0", "--dp-root", DP_ROOT, "--cell", str(y), "--out", str(out)])
    assert r.returncode == 0 and "FIXEDSTEP DONE" in r.stdout, r.stderr[-3000:]
    done = json.loads((out / "checkpoints" / "final.done").read_text())
    assert done["global_step"] == 3 and done["finished"] and len(done["checkpoint_sha256"]) == 64
    code = ("import pathlib, sys; from exp.dp_nfe.eval_dp_steps_v2 import load_workspace; "
            f"ws, cfg, pol, ident = load_workspace(pathlib.Path({DP_ROOT!r}), pathlib.Path({str(out / 'checkpoints' / 'final.ckpt')!r}), pathlib.Path({str(tmp_path / 'ev')!r})); "
            "print('LOADED', ident['cell_id'], ident['budget_steps']); "
            "from omegaconf import OmegaConf; from exp.dp_nfe.analysis.dispersion_index import sample_from_checkpoint; "
            f"sample_from_checkpoint({DP_ROOT!r}, {str(out / 'checkpoints' / 'final.ckpt')!r}, OmegaConf.to_container(cfg.task.dataset, resolve=True), {str(tmp_path / 'diagnostic.json')!r}, n_histories=2, n_samples=3, k=1, device='cpu', trainpool_cfg=OmegaConf.to_container(cfg.task.dataset, resolve=True))")
    r2 = subprocess.run([sys.executable, "-c", code], cwd=DP_ROOT, env={**os.environ, "PYTHONPATH": f"{OPENPI}:{DP_ROOT}:" + os.environ.get("PYTHONPATH", "")}, capture_output=True, text=True)
    assert r2.returncode == 0 and "LOADED pusht_lowdim_full_epsilon_s42_Bt 3" in r2.stdout, r2.stderr[-2000:]


def _official_schedulers(prediction_type):
    ddpm = diffusers.DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                                   variance_type="fixed_small", clip_sample=True, prediction_type=prediction_type)
    ddim = diffusers.DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                                   clip_sample=True, set_alpha_to_one=True, steps_offset=0, prediction_type=prediction_type)
    return ddpm, ddim


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_ddpm_step_matches_real_diffusers_scheduler(prediction_type):
    ddpm, _ = _official_schedulers(prediction_type)
    ac = ddpm.alphas_cumprod.double().tolist(); betas = ddpm.betas.double().tolist()
    g = torch.Generator().manual_seed(0)
    for t in (99, 50, 3, 1, 0):
        x = torch.randn(2, 4, 3, generator=g, dtype=torch.float64); pred = torch.randn(2, 4, 3, generator=g, dtype=torch.float64) * 0.5
        noise = torch.randn(2, 4, 3, generator=g, dtype=torch.float64)
        # diffusers draws its own posterior noise; give it a generator whose first draw is `noise`
        seed = 1234 + t
        gen = torch.Generator().manual_seed(seed); ref_noise = torch.randn(2, 4, 3, generator=gen, dtype=torch.float64)
        ref = ddpm.step(pred, t, x, generator=torch.Generator().manual_seed(seed)).prev_sample
        out = S.ddpm_step(x, pred, t, betas, ac, prediction_type, True, ref_noise)
        assert torch.allclose(out, ref, atol=1e-6), (prediction_type, t)  # diffusers keeps its coefficients in float32


def test_ddim_100_chain_matches_real_diffusers_scheduler_eps_head():
    _, ddim = _official_schedulers("epsilon")
    ddim.set_timesteps(100)
    assert list(ddim.timesteps) == S.make_timesteps(100, 100)  # leading == trailing at k = T
    ac = ddim.alphas_cumprod.double().tolist()
    W = torch.randn(6, 6, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.3
    def net(x, t):
        return torch.tanh(x @ W) * 0.8 + 0.05 * float(t) / 100
    x0 = torch.randn(2, 6, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    for use_clipped, mode in ((False, "raw"), (True, "recompute")):
        ref = x0.clone()
        for t in ddim.timesteps:
            ref = ddim.step(net(ref, int(t)), int(t), ref, eta=0.0, use_clipped_model_output=use_clipped).prev_sample
        out, nfe = S.sample_ddim(net, x0.clone(), S.make_timesteps(100, 100), ac, "epsilon", clip=True, eps_mode=mode)
        assert nfe == 100 and torch.allclose(out, ref, atol=1e-5), mode  # float32 coefficients over 100 steps


@pytest.mark.skipif(not os.environ.get("X0_PUSHT_ZARR"), reason="X0_PUSHT_ZARR (official pusht replay zarr with data/img) not set")
def test_image_cell_two_updates_on_real_pusht_zarr(tmp_path):
    """The hybrid (image) workspace config trains two fixed updates on CPU from the real PushT replay buffer (the
    image dataset returns ``obs`` as a dict), saves final.ckpt and reloads through the evaluator's loader."""
    z = os.environ["X0_PUSHT_ZARR"]
    y = _cell_yaml(tmp_path / "img.yaml", "pusht_image_full_sample_s42_Bt", "train_diffusion_unet_hybrid_workspace", "pusht_image",
                   [f"task.dataset.zarr_path={z}", "training.device=cpu"], head="sample", budget=2)
    out = tmp_path / "run_img"
    r = _fresh(["exp.dp_nfe.train_x0", "--dp-root", DP_ROOT, "--cell", str(y), "--out", str(out)])
    assert r.returncode == 0 and "FIXEDSTEP DONE" in r.stdout, r.stderr[-3000:]
    ident = json.loads((out / "identity.json").read_text())
    assert ident["modality"] == "image" and ident["head"] == "sample" and ident["n_windows"] > 1000
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert [x["step"] for x in log] == [1, 2] and all(np.isfinite(x["loss"]) for x in log)
    code = ("import pathlib; from exp.dp_nfe.eval_dp_steps_v2 import load_workspace; "
            f"ws, cfg, pol, ident = load_workspace(pathlib.Path({DP_ROOT!r}), pathlib.Path({str(out / 'checkpoints' / 'final.ckpt')!r}), pathlib.Path({str(tmp_path / 'ev')!r})); "
            "print('LOADED', ident['cell_id'], ident['budget_steps'], type(pol).__name__); "
            "from omegaconf import OmegaConf; from exp.dp_nfe.analysis.dispersion_index import sample_from_checkpoint; "
            f"sample_from_checkpoint({DP_ROOT!r}, {str(out / 'checkpoints' / 'final.ckpt')!r}, OmegaConf.to_container(cfg.task.dataset, resolve=True), {str(tmp_path / 'image_diagnostic.json')!r}, n_histories=2, n_samples=3, k=1, device='cpu')")
    r2 = subprocess.run([sys.executable, "-c", code], cwd=DP_ROOT, env={**os.environ, "PYTHONPATH": f"{OPENPI}:{DP_ROOT}:" + os.environ.get("PYTHONPATH", "")}, capture_output=True, text=True)
    assert r2.returncode == 0 and "LOADED pusht_image_full_sample_s42_Bt 2 DiffusionUnetHybridImagePolicy" in r2.stdout, r2.stderr[-2000:]


def test_image_training_and_diagnostics_with_synthetic_camera_data(tmp_path, monkeypatch):
    from exp.dp_nfe.mode_filter_datasets import _zarr_put
    import zarr
    path = tmp_path / "image_fixture.zarr"
    _synthetic_pusht_zarr(path, 128, 16, 0)
    data = zarr.open(str(path), mode="a")["data"]
    _zarr_put(data, "img", np.zeros((2048, 96, 96, 3), dtype=np.uint8))
    monkeypatch.setenv("X0_PUSHT_ZARR", str(path))
    test_image_cell_two_updates_on_real_pusht_zarr(tmp_path)
