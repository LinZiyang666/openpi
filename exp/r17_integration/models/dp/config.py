"""CPU-safe loader options; runtime value types come from P1 contracts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadSpec:
    checkpoint: str
    domain: str
    text_encoder: str = "/data/mw_finetune/dp/clip_text"
    device: str = "cpu"
    diagnostic: bool = True
    expected_weight_sha256: str | None = None
    expected_norm_identity: str | None = None
    expected_model_identity: str | None = None
    sampler_kind: str = "ddpm"
    ddim_eta: float = 0.0
    libero_benchmark_specs: list[dict] | None = None
