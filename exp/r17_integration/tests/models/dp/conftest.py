import json
from pathlib import Path
import numpy as np
import pytest
import torch

from exp.mw_finetune.dp.model import DiffusionPolicy
from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.models.dp.staged import DPAdapter


@pytest.fixture
def make_adapter():
    def make(domain="metaworld", second_camera="zero", libero_benchmark_specs=None):
        # E's actual vision/preprocessing/module implementation; only the
        # expensive Transformer width/depth and text size are reduced here.
        cfg = json.loads((Path(__file__).parents[3] / "models/dp/samples/audited_libero_metadata.json").read_text())["config"]["model"]
        cfg.update(action_dim=7 if domain == "libero" else 4,
                   lowdims=[{"key": "state", "dim": 8 if domain == "libero" else 4}],
                   lang_emb_dim=8, n_layer=1, n_cond_layers=1, n_head=2, n_emb=16,
                   p_drop_emb=0.0, p_drop_attn=0.0)
        torch.manual_seed(32)
        model = DiffusionPolicy(cfg).eval()
        with torch.no_grad():
            for name, params in model._policy.normalizer.params_dict.items():
                params["scale"].fill_(0.8 if name == "action" else 1.2)
                params["offset"].fill_(0.3 if name == "action" else -0.1)
        def language(prompts):
            return torch.tensor([[sum(p.encode()) % 7 / 7 + i / 8 for i in range(8)] for p in prompts])
        stats = {name: {"q01": [-2.0-i for i in range(dim)], "q99": [3.0+i for i in range(dim)]}
                 for name, dim in (("actions", cfg["action_dim"]), ("state", cfg["lowdims"][0]["dim"]))}
        return DPAdapter(model, stats, language, domain=domain, weight_identity="toy-only-not-475",
                         second_camera=second_camera, language_identity="toy-projected",
                         libero_benchmark_specs=libero_benchmark_specs)
    return make


@pytest.fixture
def req_context():
    return RequestContext("p4_cpu", "mt50_seed7_corner2_r17_v1", "reach-v3", 0, 0)


def observation(domain="metaworld", prompt="reach the object"):
    side = 96 if domain == "libero" else None
    image = np.arange(96*96*3, dtype=np.uint8).reshape(96, 96, 3) if side else np.arange(99*107*3, dtype=np.uint8).reshape(99, 107, 3)
    if domain == "libero":
        return {"observation/image": image, "observation/wrist_image": image[::-1].copy(),
                "observation/state": np.linspace(-0.2, 0.3, 8, dtype=np.float32), "prompt": prompt}
    return {"image": image, "state": np.linspace(-0.2, 0.3, 4, dtype=np.float32), "prompt": prompt}


def episode_metadata(task_name="reach"):
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    names = [row[0] for row in FORMAL_SPEC.tasks]
    return {"campaign": "p4_cpu", "task_name": task_name,
            "task_id": names.index(task_name) if task_name in names else 0,
            "original_init_idx": 0, "seed": 7, "explicit_seed": 18,
            "ack_protocol": "r17.execution.v1", "arm": "full"}


def canonical_recipe_options(adapter):
    from exp.r17_integration.common.benchmarks import get_benchmark, task_table_identity
    protocol = adapter.metadata().allowed_benchmark_protocols[0]
    return {"ordered_tasks": list(get_benchmark(protocol).ordered_tasks),
            "task_table_identity": task_table_identity(protocol)}


def seal_test_library(path, adapter, library, *, task_name="reach"):
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.store import write_store
    from exp.r17_integration.tests.debug.fixtures import record
    rows = [record(init=10+int(ep), seq=int(step), task=task_name, n_decisions=4)
            for ep, step in zip(library.episode, library.step)]
    arrays = {"key_"+name: list(values) for name, values in library.features.items()}
    arrays.update(state=list(library.state), normalized_actions=list(library.action))
    return write_store(path, model_binding(adapter.metadata(), adapter.metadata().allowed_benchmark_protocols[0]), rows, arrays)
