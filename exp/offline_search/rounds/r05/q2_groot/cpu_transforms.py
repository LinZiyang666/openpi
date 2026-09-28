"""Actual GR00T action-only inverse transforms, without a VLM processor/model.

LiberoDataConfig uses these three inverse transforms; GR00TTransform.unapply is
identity and video/state transforms ignore action keys. Only dependency paths
are extended: this still runs the repository's Python and torch on CPU.
"""
import os
import sys
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.extend(['/home/weiland/projects/openpi_ext/third_party/gr00t_n15',
                 '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/lib/python3.11/site-packages'])
import json
from pathlib import Path
import numpy as np
import torch
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from exp.libero_groot.policy_adapter import validate_action_chunk, chunk_to_libero_actions


def action_transform(suite):
    keys=['action.'+k for k in ('x','y','z','roll','pitch','yaw','gripper')]
    t=ComposedModalityTransform(transforms=[StateActionToTensor(apply_to=keys),
        StateActionTransform(apply_to=keys,normalization_modes={k:'min_max' for k in keys}),
        ConcatTransform(video_concat_order=[],action_concat_order=keys)])
    path=Path('/data/ckpt')/('n15_libero_spatial' if suite=='spatial' else 'n15_libero_10')/'experiment_cfg/metadata.json'
    t.set_metadata(DatasetMetadata.model_validate(json.loads(path.read_text())['new_embodiment']))
    t.eval()
    return t,path


def wire(t,chunk):
    raw=t.unapply({'action':torch.from_numpy(np.array(chunk,np.float32))[None]})
    return chunk_to_libero_actions(validate_action_chunk({k:v.squeeze(0) for k,v in raw.items()}))
