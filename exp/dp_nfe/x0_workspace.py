"""Fixed-update-budget training workspace for Diffusion Policy (runs inside the official DP env: py3.9, torch 1.12).

The official ``TrainDiffusionUnet{Lowdim,Hybrid}Workspace`` trains by epochs over the sliding-window dataset (so two datasets
with the same number of episodes but different lengths get different update counts), runs env rollouts every
``rollout_every`` epochs (0 divides by zero, ``env_runner=null`` fails an assert) and keeps top-k checkpoints by the rollout
score. This workspace keeps the official policy / dataset / normaliser / EMA / optimizer objects and replaces only the loop:

* exactly ``B = cfg.x0.budget_steps`` optimizer updates, each doing optimizer.step + LR.step + EMA.step once;
* windows are drawn with replacement, ``batch_size`` per update, from a per-step RNG ``default_rng(window_seed + step)`` so
  resuming at ``global_step`` reproduces the same batches; no epochs, no dataloader;
* LR schedule (``cfg.training.lr_scheduler``) sized by ``B`` with ``warmup = min(500, B // 10)``;
* val MSE every ``cfg.x0.val_every`` updates on a fixed held-out batch with a fixed torch RNG state (fixed t / noise);
* no env runner is built; ``latest.ckpt`` every ``cfg.x0.save_every`` updates and ``final.ckpt`` after update ``B``,
  both written to a temp file and atomically renamed; the payload carries the RNG states, the identity and the step;
* the normaliser is **not** fitted on the cell's subset: U/M cells load the task's frozen training-pool normaliser
  (``cfg.x0.normalizer_path`` + ``normalizer_sha256``, see ``x0_normalizer.py``) and refuse a hash mismatch; only cells
  without a frozen file (explore / tests) fit their own, recorded as ``normalizer_source=fitted``;
* resuming refuses a payload without an identity or whose identity differs in **any** field (data / normaliser /
  resolved-config / code / dependency hashes included); the saved RNG state is restored only after every initialisation
  (dataset, normaliser, held-out batch, optimizer placement) so the resumed trajectory equals the uninterrupted one.

Config additions (``cfg.x0``): ``cell_id, budget_steps, batch_size, window_seed, val_every, save_every, val_batch_size,
heldout_dataset (hydra target, optional), normalizer_path / normalizer_sha256 (optional), identity (dict)``. Everything
else is the official config with ``policy.noise_scheduler.prediction_type`` set to ``epsilon`` or ``sample`` and
``task.dataset.val_ratio=0``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from exp.dp_nfe.x0_identity import identity_diff, sha256_file, verify_data_identity

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def window_indices(n_windows: int, batch_size: int, window_seed: int, step: int) -> np.ndarray:
    """The batch of window indices for update ``step`` (with replacement; a pure function of (seed, step))."""
    rng = np.random.default_rng(int(window_seed) + int(step))
    return rng.integers(0, n_windows, size=batch_size)


def torch_rng_state() -> Dict[str, Any]:
    """Snapshot of the torch CPU / CUDA, numpy and python RNG states."""
    st = {"cpu": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_torch_rng_state(st: Dict[str, Any]) -> None:
    """Restore a snapshot taken by :func:`torch_rng_state`."""
    torch.set_rng_state(st["cpu"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


class FixedStepWorkspace(BaseWorkspace):
    """Fixed-update-budget DP training workspace (see the module docstring for the loop contract).

    ``cfg`` is a composed official DP config with the ``x0`` block; ``output_dir`` receives ``checkpoints/``,
    ``train_log.jsonl`` and ``identity.json``. ``run()`` returns the path of ``final.ckpt``. Raises ``ValueError`` on a
    non-positive budget, on a resume payload without / with a conflicting identity, on a frozen-normaliser hash mismatch
    and when a U/M cell has no frozen normaliser; ``RuntimeError`` on a non-finite loss or gradient."""

    include_keys = ("global_step", "samples_seen", "identity", "rng_state", "train_log", "finished", "ema_state")
    # identity fields compared on resume are *all* fields; these are derived from the config here, the rest come from
    # the cell (train_x0.compose_cell adds the data / normaliser / config / code / dependency hashes)
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir=output_dir)
        x0 = cfg.x0
        self.budget = int(x0.budget_steps)
        if self.budget <= 0:
            raise ValueError("cfg.x0.budget_steps must be positive")
        _seed_all(int(cfg.training.seed))
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        warmup = min(500, self.budget // 10)
        self.lr_scheduler = get_scheduler(cfg.training.lr_scheduler, optimizer=self.optimizer,
                                          num_warmup_steps=warmup, num_training_steps=self.budget, last_epoch=-1)
        self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        self.global_step = 0
        self.samples_seen = 0
        self.finished = False
        self.train_log = []
        self.rng_state = None
        self._pending_rng = None
        self.ema_state = None  # DP's EMAModel is not a Module: its step counter / decay are pickled explicitly
        self.normalizer_path = str(x0.normalizer_path) if x0.get("normalizer_path") else None
        self.normalizer_sha256 = str(x0.normalizer_sha256) if x0.get("normalizer_sha256") else None
        self.identity = {}
        if "identity" in x0 and x0.identity is not None:
            self.identity.update(OmegaConf.to_container(x0.identity, resolve=True))
        self.identity.update({"cell_id": str(x0.cell_id), "budget_steps": self.budget, "batch_size": int(x0.batch_size),
                              "window_seed": int(x0.window_seed), "train_seed": int(cfg.training.seed),
                              "head": str(cfg.policy.noise_scheduler.prediction_type), "task": str(cfg.task.name),
                              "warmup_steps": warmup, "normalizer_sha256": self.normalizer_sha256})
        if self.identity.get("variant") in ("U", "M") and not (self.normalizer_path and self.normalizer_sha256):
            raise ValueError(f"cell {x0.cell_id}: U/M cells must load the frozen training-pool normalizer "
                             f"(cfg.x0.normalizer_path / normalizer_sha256 missing)")

    # ------------------------------------------------------------------ checkpoints
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        """Restore a payload written by :meth:`save_atomic`. Refuses (``ValueError``) a payload without an identity or
        whose identity differs from this workspace's in any field; the RNG state is *deferred* to :meth:`run` (applied
        after all initialisation) -- callers that only need the weights (the evaluator) never apply it."""
        import dill
        if "identity" not in payload.get("pickles", {}):
            raise ValueError("checkpoint carries no identity; refusing to resume from it")
        ident = dill.loads(payload["pickles"]["identity"])
        diff = identity_diff(ident, self.identity)
        if diff:
            raise ValueError("checkpoint identity mismatch: " + "; ".join(diff))
        super().load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        self._pending_rng = self.rng_state
        if self.ema_state is not None:
            self.ema.optimization_step = int(self.ema_state["optimization_step"])
            self.ema.decay = float(self.ema_state["decay"])

    def save_atomic(self, tag: str) -> pathlib.Path:
        """Write ``checkpoints/<tag>.ckpt`` (tmp file + rename) with the RNG / EMA state captured now, plus ``<tag>.done`` (step, identity, finished; the final tag also records the checkpoint sha256)."""
        path = pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rng_state = torch_rng_state()
        self.ema_state = {"optimization_step": int(self.ema.optimization_step), "decay": float(self.ema.decay)}
        tmp = path.with_suffix(".ckpt.tmp")
        self.save_checkpoint(path=str(tmp), use_thread=False)
        os.replace(tmp, path)
        done = {"global_step": self.global_step, "identity": self.identity, "finished": bool(self.finished)}
        if tag == "final":
            done["checkpoint_sha256"] = sha256_file(path)
        (path.parent / f"{tag}.done").write_text(json.dumps(done, indent=1))
        return path

    # ------------------------------------------------------------------ training
    def _fixed_val_batch(self, cfg, device):
        n = int(cfg.x0.val_batch_size)
        if cfg.x0.get("heldout_dataset") is not None:
            ds = hydra.utils.instantiate(cfg.x0.heldout_dataset)
            self.heldout_windows = len(ds)
        else:
            ds = self.dataset
            self.heldout_windows = 0  # marked: validation on training windows
        rng = np.random.default_rng(int(cfg.x0.window_seed) + 10_000_003)
        idx = rng.integers(0, len(ds), size=min(n, len(ds)))
        batch = default_collate([ds[int(i)] for i in idx])
        return dict_apply(batch, lambda x: x.to(device))

    @torch.no_grad()
    def _val_mse(self, batch) -> float:
        st = torch_rng_state()
        torch.manual_seed(int(self.cfg.x0.window_seed) + 7)  # fixed t / noise for every evaluation
        self.ema_model.eval()
        loss = float(self.ema_model.compute_loss(batch).item())
        self.ema_model.train()
        set_torch_rng_state(st)
        if not math.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        return loss

    def run(self):
        """Resume if ``checkpoints/latest.ckpt`` exists, build the dataset / normaliser / held-out batch, restore the RNG state, run exactly ``budget`` updates and return the path of ``final.ckpt``."""
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        out = pathlib.Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        verify_data_identity(self.identity)
        if cfg.training.get("resume", True):
            final = out / "checkpoints" / "final.ckpt"
            done = out / "checkpoints" / "final.done"
            if final.is_file() and done.is_file():
                record = json.loads(done.read_text())
                if (record.get("finished") is True and record.get("global_step") == self.budget
                        and not identity_diff(record.get("identity", {}), self.identity)
                        and sha256_file(final) == record.get("checkpoint_sha256")):
                    self.load_checkpoint(path=str(final))
                    return final
            latest = out / "checkpoints" / "latest.ckpt"
            if latest.is_file():
                print(f"resuming from {latest}")
                self.load_checkpoint(path=str(latest))
        self.dataset = hydra.utils.instantiate(cfg.task.dataset)
        n_windows = len(self.dataset)
        if self.normalizer_path:
            from exp.dp_nfe.x0_normalizer import load_frozen, normalizer_sha256
            normalizer = load_frozen(pathlib.Path(self.normalizer_path), self.normalizer_sha256)
            normalizer_source = "frozen"
        else:
            from exp.dp_nfe.x0_normalizer import normalizer_sha256
            normalizer = self.dataset.get_normalizer()
            normalizer_source = "fitted"
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)
        loaded_sha = normalizer_sha256(self.model.normalizer.state_dict())
        if self.normalizer_sha256 and loaded_sha != self.normalizer_sha256:
            raise ValueError(f"normalizer installed in the policy ({loaded_sha}) != frozen ({self.normalizer_sha256})")
        self.model.to(device); self.ema_model.to(device)
        from diffusion_policy.common.pytorch_util import optimizer_to
        optimizer_to(self.optimizer, device)
        val_batch = self._fixed_val_batch(cfg, device)
        bs = int(cfg.x0.batch_size)
        log_path = out / "train_log.jsonl"
        ident_path = out / "identity.json"
        ident_now = {**self.identity, "n_windows": n_windows, "heldout_windows": self.heldout_windows,
                     "normalizer_source": normalizer_source, "normalizer_sha256_loaded": loaded_sha}
        if ident_path.is_file():
            prev = json.loads(ident_path.read_text())
            diff = identity_diff(prev, ident_now)
            if diff:
                raise ValueError(f"{ident_path} was written by a different cell: " + "; ".join(diff))
        else:
            ident_path.write_text(json.dumps(ident_now, indent=1))
        # the resumed RNG state is applied only now, after every initialisation that could consume random numbers
        if self._pending_rng is not None:
            set_torch_rng_state(self._pending_rng)
            self._pending_rng = None
        self.model.train()
        t0 = time.time()
        with open(log_path, "a") as lf:
            while self.global_step < self.budget:
                step = self.global_step
                idx = window_indices(n_windows, bs, int(cfg.x0.window_seed), step)
                batch = default_collate([self.dataset[int(i)] for i in idx])
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = self.model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at update {step}")
                loss.backward()
                grads_ok = all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
                if not grads_ok:
                    raise RuntimeError(f"non-finite gradient at update {step}")
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.step(self.model)
                self.global_step += 1
                self.samples_seen += bs
                rec = {"step": self.global_step, "loss": float(loss.item()), "lr": float(self.lr_scheduler.get_last_lr()[0]),
                       "elapsed_s": round(time.time() - t0, 1)}
                if self.global_step % int(cfg.x0.val_every) == 0 or self.global_step == self.budget:
                    rec["val_mse_ema"] = self._val_mse(val_batch)
                    self.train_log.append(rec)
                lf.write(json.dumps(rec) + "\n"); lf.flush()
                if self.global_step % int(cfg.x0.save_every) == 0 and self.global_step < self.budget:
                    self.save_atomic("latest")
        self.finished = True
        path = self.save_atomic("final")
        self.save_atomic("latest")
        print(f"FIXEDSTEP DONE cell={self.identity['cell_id']} steps={self.global_step} ema_steps={self.ema.optimization_step} "
              f"samples_seen={self.samples_seen} final={path}")
        return path
