"""RLinf π0.5 MetaWorld SFT adapter; uses the existing model and normalization."""
from exp.mw_finetune.adapters import canonical_to_wire, validate_actions


class Pi05Policy:
    def __init__(self, policy, denoising_steps):
        self.policy = policy
        self.denoising_steps = denoising_steps

    def reset(self):
        pass  # Existing openpi PyTorch inference has no history/action queue.

    def infer(self, obs):
        import numpy as np
        noise = np.random.default_rng(obs["seed"]).standard_normal((5, 32)).astype(np.float32)
        return {"actions": validate_actions(self.policy.infer(canonical_to_wire(obs), noise=noise)["actions"])}

    def get_metadata(self):
        return {"config": "pi05_metaworld", "checkpoint_source": "RLinf/RLinf-Pi05-MetaWorld-SFT",
                "denoising_steps": self.denoising_steps, "action_horizon": 5,
                "noise": "NumPy PCG64 standard_normal; episode/decision seed",
                "normalization": "existing checkpoint assets/metaworld_mt50 quantile stats",
                "cache": False, "warm_start": False}


def load_policy(checkpoint, device="cuda:0", **kw):
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    steps = kw.pop("denoising_steps", 10)
    if kw:
        raise ValueError(f"unknown π0.5 options: {sorted(kw)}")
    policy = create_trained_policy(get_config("pi05_metaworld"), checkpoint,
                                   sample_kwargs={"num_steps": steps}, pytorch_device=device)
    return Pi05Policy(policy, steps)
