"""Shared, frozen normaliser of one (task, modality) of the x0-head experiment (plan §3.2 / G2 R1-B2).

The four cells of a task (U/M x epsilon/sample) must see identical action / observation scaling, otherwise the U-vs-M
intervention also changes the clip boundary and the noise difficulty. The normaliser is therefore fitted **once** on the
task's training pool (every non-held-out episode, ``<task>_trainpool<ext>`` from ``mode_filter_datasets.py``) with the
same dataset class the cells use, written to ``<subsets>/<task>_<modality>_normalizer.pt`` with a ``.json`` side-car
carrying its canonical hash, and every cell loads that file and verifies the hash (``x0_workspace.FixedStepWorkspace``).

Inside the DP env::

    PYTHONPATH=<openpi root>:<dp root> python -m exp.dp_nfe.x0_normalizer --dp-root <dp root> \
        --workspace-config train_diffusion_unet_lowdim_workspace --task pusht_lowdim \
        --override task.dataset.zarr_path=<subsets>/pusht_trainpool.zarr --out <subsets>/pusht_lowdim_normalizer.pt

The hash is computed over the state_dict tensors (sorted keys, dtype, shape, little-endian bytes), not over the torch
file, so it is stable across torch serialisation details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Dict, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def normalizer_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Canonical hash of a normaliser state_dict: sorted keys, dtype, shape and the float bytes of each tensor."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        v = state_dict[k]
        arr = np.ascontiguousarray(v.detach().cpu().numpy())
        h.update(k.encode()); h.update(str(arr.dtype).encode()); h.update(str(arr.shape).encode())
        h.update(arr.astype(arr.dtype.newbyteorder("<"), copy=False).tobytes())
    return h.hexdigest()


def save_normalizer(normalizer, path: pathlib.Path, source: Dict[str, object]) -> Dict[str, object]:
    """Write ``path`` (torch state_dict) and ``path.json`` ({sha256, keys, source}); returns the side-car dict."""
    sd = {k: v.detach().cpu() for k, v in normalizer.state_dict().items()}
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, str(path))
    side = {"sha256": normalizer_sha256(sd), "keys": sorted(sd.keys()), "source": source, "file": path.name}
    pathlib.Path(str(path) + ".json").write_text(json.dumps(side, indent=1, default=str))
    return side


def read_sidecar(path: pathlib.Path) -> Dict[str, object]:
    """The ``.json`` side-car of a frozen normaliser file (raises FileNotFoundError when it is missing)."""
    return json.loads(pathlib.Path(str(path) + ".json").read_text())


def load_frozen(path: pathlib.Path, expected_sha256: str):
    """Load a frozen normaliser as a DP ``LinearNormalizer`` and verify its canonical hash.

    Raises ``FileNotFoundError`` when the file or its side-car is missing and ``ValueError`` when the recomputed hash
    differs from ``expected_sha256`` or from the side-car."""
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    path = pathlib.Path(path)
    side = read_sidecar(path)
    sd = torch.load(str(path), map_location="cpu")
    got = normalizer_sha256(sd)
    if got != expected_sha256 or got != side["sha256"]:
        raise ValueError(f"frozen normalizer hash mismatch for {path}: file={got} sidecar={side['sha256']} "
                         f"expected={expected_sha256}")
    n = LinearNormalizer()
    n.load_state_dict(sd)
    return n


def fit_from_config(dp_root: pathlib.Path, workspace_config: str, task: str, overrides, out: pathlib.Path) -> Dict[str, object]:
    """Compose the official config (``task=<task>`` + overrides), instantiate ``cfg.task.dataset``, fit its normaliser
    and freeze it at ``out``. Returns the side-car dict. Must run inside the DP env."""
    import hydra
    from hydra import compose, initialize_config_dir
    config_dir = dp_root / "diffusion_policy" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=workspace_config, overrides=[f"task={task}", "task.dataset.val_ratio=0"] + list(overrides))
    if "max_train_episodes" in cfg.task.dataset:
        cfg.task.dataset.max_train_episodes = None
    ds_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
    if ds_cfg.get("val_ratio") != 0:
        raise ValueError("the shared training-pool normalizer requires val_ratio=0")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    src_path = None
    for key in ("zarr_path", "dataset_path", "dataset_dir"):
        if key in ds_cfg:
            src_path = ds_cfg[key]
    source = {"workspace_config": workspace_config, "task": task, "overrides": list(overrides), "dataset_cfg": ds_cfg,
              "dataset_path": src_path, "n_windows": len(dataset), "torch": torch.__version__}
    from exp.dp_nfe.x0_identity import sha256_path
    source["dataset_sha256"] = sha256_path(pathlib.Path(src_path))
    return save_normalizer(normalizer, out, source)


def main() -> None:
    """CLI entry: fit + freeze one (task, modality) normaliser from the training pool."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dp-root", required=True)
    ap.add_argument("--workspace-config", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dp_root = pathlib.Path(a.dp_root).resolve()
    sys.path.insert(0, str(dp_root))
    side = fit_from_config(dp_root, a.workspace_config, a.task, a.override, pathlib.Path(a.out))
    print(f"NORMALIZER FROZEN {a.out} sha256={side['sha256']} keys={len(side['keys'])}")


if __name__ == "__main__":
    main()
