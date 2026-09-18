"""Build one training cell of the x0-head experiment from the official Diffusion Policy configs and run it with the
fixed-update-budget workspace (``exp/dp_nfe/x0_workspace.py``). Runs inside the official DP env::

    PYTHONPATH=<openpi root>:<dp root> python -m exp.dp_nfe.train_x0 --dp-root <dp root> \
        --cell exp/dp_nfe/config/x0_multimodal/cells/pusht_lowdim_U_sample_s42_B50k.yaml --out /data/dp/x0_multimodal/runs/<cell_id>

Cell yaml keys (written by ``x0_cells.py``): ``cell_id, workspace_config, task, head (epsilon|sample), train_seed,
budget_steps, batch_size, window_seed, val_every, save_every, val_batch_size, overrides (hydra overrides, e.g. the subset
path), heldout_path_key, heldout_path (optional: the common held-out subset in the task's native format),
normalizer_path, normalizer_sha256 (the frozen training-pool normaliser; mandatory for U/M cells), identity (dict)``.

The composed, resolved config is written to ``<out>/resolved_config.yaml`` and the cell identity -- the cell's own fields
plus the hashes of the resolved config, of the training code (``x0_workspace.py``, ``train_x0.py``, ``x0_normalizer.py``, ``x0_identity.py``),
of the DP revision and of the dependency versions -- to ``<out>/manifest.json`` **before** training starts. An existing
manifest is never overwritten: a launch whose identity differs from the one already in ``<out>`` is refused, so a cell
directory can only ever hold one (data, normaliser, config, code) combination.

``${eval:...}`` (used by the official dataset / policy configs) is registered here, before any compose/resolve, so a fresh
process reaches model creation without importing the workspace first (G2 R1-B1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import yaml
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

CODE_FILES = ("x0_workspace.py", "train_x0.py", "x0_normalizer.py", "x0_identity.py")


def _git_rev(path: pathlib.Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "diffusers", "hydra", "numpy", "zarr", "h5py"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    return out


def code_sha256() -> str:
    """sha256 over the bytes of the training code files (``CODE_FILES`` next to this module), in order."""
    h = hashlib.sha256()
    here = pathlib.Path(__file__).resolve().parent
    for name in CODE_FILES:
        h.update(name.encode()); h.update((here / name).read_bytes())
    return h.hexdigest()


def compose_cell(dp_root: pathlib.Path, cell: dict, out_dir: pathlib.Path, budget_override: int = 0) -> OmegaConf:
    """Compose the official workspace config of ``cell`` (task / seed / head / val_ratio=0 / EMA / offline logging +
    the cell's overrides), retarget it to ``FixedStepWorkspace`` and attach the ``x0`` block (budget, batch, window seed,
    validation cadence, optional held-out dataset config, frozen-normaliser path + hash, identity).

    ``budget_override`` (pilots) replaces ``budget_steps`` and is recorded as ``identity.budget_override``. Returns the
    (unresolved) OmegaConf; ``identity.resolved_config_sha256`` is filled in by :func:`build_identity`. Raises
    ``ValueError`` on a head other than epsilon|sample and when a U/M cell lacks ``normalizer_path``/``normalizer_sha256``."""
    from hydra import compose, initialize_config_dir
    config_dir = dp_root / "diffusion_policy" / "config"
    head = cell["head"]
    if head not in ("epsilon", "sample"):
        raise ValueError(f"head must be epsilon|sample, got {head}")
    identity = dict(cell.get("identity", {}))
    if identity.get("variant") in ("U", "M") and not (cell.get("normalizer_path") and cell.get("normalizer_sha256")):
        raise ValueError(f"cell {cell['cell_id']}: U/M cells need normalizer_path + normalizer_sha256")
    overrides = [f"task={cell['task']}", f"training.seed={int(cell['train_seed'])}",
                 f"policy.noise_scheduler.prediction_type={head}", "task.dataset.val_ratio=0",
                 "training.use_ema=True", "logging.mode=offline"] + list(cell.get("overrides", []))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=cell["workspace_config"], overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    # The exported subset is the sampling population; upstream PushT configs otherwise cap it at 90 episodes.
    cfg.task.dataset.val_ratio = 0
    if "max_train_episodes" in cfg.task.dataset:
        cfg.task.dataset.max_train_episodes = None
    cfg._target_ = "exp.dp_nfe.x0_workspace.FixedStepWorkspace"
    # the official configs stamp ${now:...} into logging / multi_run names: pin them so the resolved config (and its
    # hash) is a pure function of the cell
    if "logging" in cfg:
        cfg.logging.name = str(cell["cell_id"])
    if "multi_run" in cfg:
        cfg.multi_run = {"run_dir": str(out_dir), "wandb_name_base": str(cell["cell_id"])}
    budget = int(budget_override or cell["budget_steps"])
    if budget_override:
        identity["budget_override"] = int(budget_override)
    x0 = {"cell_id": cell["cell_id"], "budget_steps": budget, "batch_size": int(cell.get("batch_size", 256)),
          "window_seed": int(cell.get("window_seed", 1000 + int(cell["train_seed"]))),
          "val_every": int(cell.get("val_every", 1000)), "save_every": int(cell.get("save_every", 2000)),
          "val_batch_size": int(cell.get("val_batch_size", 256)), "identity": identity,
          "normalizer_path": cell.get("normalizer_path"), "normalizer_sha256": cell.get("normalizer_sha256"),
          "heldout_dataset": None}
    if cell.get("heldout_path"):
        hd = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        hd[cell["heldout_path_key"]] = cell["heldout_path"]
        hd["val_ratio"] = 0
        x0["heldout_dataset"] = hd
    cfg.x0 = OmegaConf.create(x0)
    cfg.hydra = {"run": {"dir": str(out_dir)}}
    return cfg


def build_identity(cfg: OmegaConf, cell: dict, cell_yaml: pathlib.Path, dp_root: pathlib.Path) -> dict:
    """Complete ``cfg.x0.identity`` in place with the provenance hashes and return the resolved config container.

    Adds ``resolved_config_sha256`` (over the resolved config with the identity block blanked, so the hash does not
    depend on itself, and without the ``hydra`` block), ``code_sha256``, ``cell_yaml_sha256``, ``dp_rev`` and ``deps``
    (package versions)."""
    from exp.dp_nfe.x0_identity import expected_identity, sha256_path, verify_data_identity
    ident = expected_identity(cell)
    ident.update(OmegaConf.to_container(cfg.x0.identity, resolve=True) or {})
    ident["budget_steps"] = int(cfg.x0.budget_steps)
    ident["warmup_steps"] = min(500, int(cfg.x0.budget_steps) // 10)
    dataset = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    declared = ident.get("subset_path")
    paths = [dataset[k] for k in ("zarr_path", "dataset_path", "dataset_dir", "path") if dataset.get(k)]
    if not declared and len(paths) == 1:
        declared = paths[0]
        ident["subset_path"] = str(declared)
    if declared:
        if not any(pathlib.Path(p).resolve() == pathlib.Path(declared).resolve() for p in paths):
            raise ValueError("composed dataset path differs from frozen subset_path")
        if not ident.get("subset_sha256"):
            ident["subset_sha256"] = sha256_path(pathlib.Path(declared))
    if ident.get("variant") in ("U", "M"):
        for name in ("subset", "heldout", "subset_manifest"):
            if not ident.get(name + "_path") or not ident.get(name + "_sha256"):
                raise ValueError(f"U/M cell missing frozen {name} path/hash")
        heldout = OmegaConf.to_container(cfg.x0.heldout_dataset, resolve=True)
        if not any(str(heldout.get(k)) == str(ident["heldout_path"]) for k in ("zarr_path", "dataset_path", "dataset_dir")):
            raise ValueError("composed held-out path differs from frozen heldout_path")
    verify_data_identity(ident)
    cfg.x0.identity = OmegaConf.create(ident)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    probe = json.loads(json.dumps(resolved, default=str))
    probe["x0"]["identity"] = None
    probe.pop("hydra", None)  # output-dir bookkeeping, not part of what is trained
    cfg_sha = hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()
    ident.update({"resolved_config_sha256": cfg_sha, "code_sha256": code_sha256(),
                  "cell_yaml_sha256": hashlib.sha256(cell_yaml.read_bytes()).hexdigest(),
                  "dp_rev": _git_rev(dp_root), "deps": json.dumps(_versions(), sort_keys=True)})
    cfg.x0.identity = OmegaConf.create(ident)
    resolved["x0"]["identity"] = dict(ident)
    return resolved


def write_manifest(out: pathlib.Path, cell: dict, cfg: OmegaConf, resolved: dict) -> dict:
    """Write ``resolved_config.yaml`` + ``manifest.json`` into ``out`` unless a manifest already exists; an existing
    manifest must carry the same identity (``SystemExit`` otherwise) and is kept byte-for-byte."""
    from exp.dp_nfe.x0_identity import identity_diff
    ident = OmegaConf.to_container(cfg.x0.identity, resolve=True)
    manifest = {"cell": cell, "identity": ident, "budget_steps": int(cfg.x0.budget_steps), "head": cell["head"],
                "task": cell["task"], "train_seed": int(cell["train_seed"])}
    mpath = out / "manifest.json"
    if mpath.is_file():
        prev = json.loads(mpath.read_text())
        diff = identity_diff(prev.get("identity", {}), ident)
        if diff:
            raise SystemExit(f"{mpath} belongs to a different cell identity; refusing to overwrite: " + "; ".join(diff))
        return prev
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    mpath.write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    """CLI entry: compose the cell, bind its identity, write / verify the manifest and train (or ``--dry-run``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dp-root", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=0, help="override budget_steps (pilot runs)")
    ap.add_argument("--dry-run", action="store_true", help="compose + write manifest, do not train")
    a = ap.parse_args()
    dp_root = pathlib.Path(a.dp_root).resolve()
    sys.path.insert(0, str(dp_root))
    cell_yaml = pathlib.Path(a.cell).resolve()
    cell = yaml.safe_load(open(cell_yaml))
    out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg = compose_cell(dp_root, cell, out, a.budget)
    resolved = build_identity(cfg, cell, cell_yaml, dp_root)
    write_manifest(out, cell, cfg, resolved)
    if a.dry_run:
        print("DRY RUN ok", out); return
    import os
    os.chdir(str(dp_root))  # DP resolves relative asset paths from its root
    from exp.dp_nfe.x0_workspace import FixedStepWorkspace
    ws = FixedStepWorkspace(cfg, output_dir=str(out))
    ws.run()


if __name__ == "__main__":
    main()
