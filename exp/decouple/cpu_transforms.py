"""Load serving CPU transforms without calling either model loader."""
from pathlib import Path
import dataclasses
import json
import sys

REPO = Path(__file__).resolve().parents[2]
ISLAND = Path("/home/weiland/projects/openpi_ext/third_party/gr00t_n15")
ISLAND_PACKAGES = Path("/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/lib/python3.11/site-packages")


def pi05():
    from openpi import transforms
    from openpi.training import config, checkpoints
    train = dataclasses.replace(config.get_config("pi05_libero"),
                                assets_base_dir="/home/weiland/projects/openpi/assets")
    data = train.data.create(train.assets_dirs, train.model)
    stats = checkpoints.load_norm_stats(train.assets_dirs, data.asset_id)
    inputs = [transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
              transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs]
    outputs = [*data.model_transforms.outputs,
               transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs]
    return inputs, outputs


def groot():
    # Caller selects the island package directory BEFORE importing numpy/torch,
    # while still using the owner's required main-tree Python executable.
    sys.path.insert(0, str(ISLAND))
    sys.path.insert(0, str(ISLAND / "examples/Libero"))
    from custom_data_config import LiberoDataConfig
    from gr00t.data.schema import DatasetMetadata
    transforms = LiberoDataConfig().transform()
    transforms.eval()
    metadata = json.loads(Path("/data/ckpt/n15_libero_spatial/experiment_cfg/metadata.json").read_text())
    # LIBERO uses the new_embodiment tag in the serving entry point.
    from exp.libero_groot.serve_groot_libero import EMBODIMENT_TAG
    key = EMBODIMENT_TAG.value if hasattr(EMBODIMENT_TAG, "value") else EMBODIMENT_TAG
    transforms.set_metadata(DatasetMetadata.model_validate(metadata[key]))
    return list(transforms.transforms), transforms
