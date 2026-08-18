"""Serve the RoboCasa365 pi0.5 teacher (converted JAX->PyTorch) for Step 0b.

Deliberately standalone: it builds the TrainConfig in-process rather than
registering one in openpi/training/config.py, because the upstream fork's
robocasa configs import `robocasa` at module level and that package only
exists inside the py3.12 sim island -- adding it to our config.py would break
config import in the main venv.
"""

import dataclasses
import logging

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config as _pc
from openpi.policies import robocasa_policy
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO)

CKPT = "/home/weiland/ckpt_pi05_robocasa_pytorch"
NS_DIR = "/home/weiland/ckpt_pi05_robocasa_pytorch/assets/robocasa"
PORT = 8010


@dataclasses.dataclass(frozen=True)
class _RobocasaGroup:
    """GroupFactory producing the fork's Robocasa input/output transforms."""

    def __call__(self, model_config):
        return transforms.Group(
            inputs=[
                robocasa_policy.RobocasaInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )


@dataclasses.dataclass(frozen=True)
class _RobocasaDataConfig(_config.SimpleDataConfig):
    """Force mean/std normalization.

    Our openpi sets use_quantile_norm = (model_type != PI0) at config.py:187,
    i.e. ON for pi05. The robocasa-benchmark fork has no such line, so this
    checkpoint was trained with plain mean/std -- and indeed its norm_stats.json
    carries q01/q99 = null. Leaving quantile norm on raises at transforms.py:458.
    """

    def create(self, assets_dirs, model_config):
        dc = super().create(assets_dirs, model_config)
        return dataclasses.replace(dc, use_quantile_norm=False)


cfg = _config.TrainConfig(
    name="robocasa_infer",
    # Must match how the checkpoint was trained: pi05_pretrain_human300 used
    # Pi0Config(pi05=True, max_token_len=200); everything else default
    # (action_dim=32, action_horizon=50, discrete_state_input=True).
    model=pi0_config.Pi0Config(pi05=True, max_token_len=200),
    data=_RobocasaDataConfig(data_transforms=_RobocasaGroup()),
)

print("loading norm stats from", NS_DIR, flush=True)
ns = _normalize.load(NS_DIR)
print("  norm stats keys:", list(ns), flush=True)

print("loading policy from", CKPT, flush=True)
policy = _pc.create_trained_policy(cfg, CKPT, norm_stats=ns, pytorch_device="cuda")
print("POLICY-READY", flush=True)

meta = policy.metadata if hasattr(policy, "metadata") else {}
server = websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=PORT, metadata=meta)
print(f"SERVER-LISTENING on 0.0.0.0:{PORT}", flush=True)
server.serve_forever()
