"""Checkpoint-exact module tree, reconstructed from metadata and 475 tensors.

Transformer code is pinned Stanford DP; vision math follows robomimic's
ResNet18ConvFiLM and SpatialSoftmax. See RECIPE for inferred wrapper conventions.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from diffusers import DDPMScheduler

from .vendor.transformer import TransformerForDiffusion


class FiLM(nn.Module):
    def __init__(self, channels, language_dim):
        super().__init__()
        self.lang_proj = nn.Linear(language_dim, channels * 2)

    def forward(self, x, language):
        beta, gamma = self.lang_proj(language).chunk(2, dim=-1)
        return F.relu(x * (1 + gamma[..., None, None]) + beta[..., None, None])


class FiLMResNet(nn.Module):
    def __init__(self, language_dim):
        super().__init__()
        net = resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
        self._base_block = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self._conv_blocks = nn.ModuleList([b for l in (net.layer1, net.layer2, net.layer3, net.layer4) for b in l])
        self._film_layers = nn.ModuleList([FiLM(b.conv2.out_channels, language_dim) for b in self._conv_blocks])

    def forward(self, image, language):
        x = self._base_block(image)
        for block, film in zip(self._conv_blocks, self._film_layers, strict=True):
            x = film(block(x), language)
        return x


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.Conv2d(512, 32, 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 3), torch.linspace(-1, 1, 3), indexing="ij")
        self.register_buffer("pos_x", x.reshape(1, -1))
        self.register_buffer("pos_y", y.reshape(1, -1))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, features):
        attention = (self.nets(features).reshape(-1, 9) / self.temperature).softmax(-1)
        xy = torch.cat([(attention * self.pos_x).sum(-1, keepdim=True),
                        (attention * self.pos_y).sum(-1, keepdim=True)], -1)
        return xy.reshape(features.shape[0], 32, 2)


class VisualCore(nn.Module):
    def __init__(self, language_dim):
        super().__init__()
        self.backbone = FiLMResNet(language_dim)
        self.nets = nn.Sequential(SpatialSoftmax(), nn.Flatten(1), nn.Linear(64, 64))

    def forward(self, image, language):
        return self.nets(self.backbone(image, language))


def crop(images, training):
    if not training:
        return images[..., 6:90, 6:90]
    # Stanford random crop has offsets in [0, H-CH-1]; one crop per image.
    offsets = torch.randint(0, 12, (len(images), 2), device=images.device)
    y = offsets[:, 0, None, None] + torch.arange(84, device=images.device)[None, :, None]
    x = offsets[:, 1, None, None] + torch.arange(84, device=images.device)[None, None, :]
    b = torch.arange(len(images), device=images.device)[:, None, None]
    return images.permute(0, 2, 3, 1)[b, y, x].permute(0, 3, 1, 2).contiguous()


class ObsEncoder(nn.Module):
    def __init__(self, language_dim, feature_order):
        super().__init__()
        self.obs_nets = nn.ModuleDict({k: VisualCore(language_dim) for k in ("base_0_rgb", "left_wrist_0_rgb")})
        self.feature_order = feature_order

    def forward(self, obs):
        # robomimic ObservationEncoder applies ReLU after each visual core.
        features = {k: F.relu(self.obs_nets[k](crop(obs[k], self.training), obs["lang_emb"])) for k in self.obs_nets}
        features.update(state=obs["state"], lang_emb=obs["lang_emb"])
        return torch.cat([features[k] for k in self.feature_order], -1).unsqueeze(1)


class LinearNormalizer(nn.Module):
    def __init__(self, action_dim, state_dim, language_dim):
        super().__init__()
        dims = {"action": action_dim, "state": state_dim, "lang_emb": language_dim,
                "base_0_rgb": 96, "left_wrist_0_rgb": 96}
        self.params_dict = nn.ModuleDict()
        for key, dim in dims.items():
            values = nn.ParameterDict({"scale": nn.Parameter(torch.ones(dim), requires_grad=False),
                                       "offset": nn.Parameter(torch.zeros(dim), requires_grad=False)})
            values["input_stats"] = nn.ParameterDict({s: nn.Parameter(torch.zeros(dim), requires_grad=False)
                                                     for s in ("min", "max", "mean", "std")})
            self.params_dict[key] = values

    def normalize(self, key, x):
        p = self.params_dict[key]
        return x * p["scale"] + p["offset"]

    def unnormalize(self, key, x):
        p = self.params_dict[key]
        return (x - p["offset"]) / p["scale"]


class Core(nn.Module):
    def __init__(self, cfg, feature_order):
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.empty(0))
        self.mask_generator = nn.Module()
        self.mask_generator.register_parameter("_dummy_variable", nn.Parameter(torch.empty(0)))
        self.obs_encoder = ObsEncoder(cfg["lang_emb_dim"], feature_order)
        self.normalizer = LinearNormalizer(cfg["action_dim"], cfg["lowdims"][0]["dim"], cfg["lang_emb_dim"])
        self.model = TransformerForDiffusion(input_dim=cfg["action_dim"], output_dim=cfg["action_dim"],
            horizon=cfg["horizon"], n_obs_steps=1, cond_dim=128 + cfg["lowdims"][0]["dim"] + cfg["lang_emb_dim"],
            **{k: cfg[k] for k in ("n_layer", "n_cond_layers", "n_head", "n_emb", "p_drop_emb", "p_drop_attn",
                                   "causal_attn", "time_as_cond", "obs_as_cond")})


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg, feature_order=("base_0_rgb", "left_wrist_0_rgb", "state", "lang_emb")):
        super().__init__()
        if cfg["n_obs_steps"] != 1 or list(cfg["crop_shape"]) != [84, 84] or cfg["horizon"] != 16:
            raise ValueError("Only the audited stateless 96/84, horizon16 architecture is supported")
        self.cfg = cfg
        self._policy = Core(cfg, feature_order)

    def scheduler(self, steps=None):
        c = self.cfg
        scheduler = DDPMScheduler(**{k: c[k] for k in ("num_train_timesteps", "beta_schedule", "beta_start",
            "beta_end", "variance_type", "prediction_type", "clip_sample", "clip_sample_range")})
        scheduler.set_timesteps(steps or c["num_inference_steps"])
        return scheduler

    def condition(self, obs):
        return self._policy.obs_encoder({k: self._policy.normalizer.normalize(k, x) for k, x in obs.items()})

    def loss(self, obs, action):
        core = self._policy
        cond = self.condition(obs)
        action = core.normalizer.normalize("action", action)
        noise = torch.randn_like(action)
        timestep = torch.randint(self.cfg["num_train_timesteps"], (len(action),), device=action.device)
        noisy = self.scheduler().add_noise(action, noise, timestep)
        return F.mse_loss(core.model(noisy, timestep, cond), noise)

    @torch.inference_mode()
    def sample(self, obs, generator=None, steps=None):
        core = self._policy
        cond = self.condition(obs)
        trajectory = torch.randn((len(cond), 16, self.cfg["action_dim"]), dtype=cond.dtype,
                                 device=cond.device, generator=generator)
        scheduler = self.scheduler(steps)
        for t in scheduler.timesteps:
            noise = core.model(trajectory, t, cond)
            trajectory = scheduler.step(noise, t, trajectory, generator=generator).prev_sample
        return core.normalizer.unnormalize("action", trajectory)
