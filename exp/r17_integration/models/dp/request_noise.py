"""Private sequential streams preserve E parity and are independent of dispatch."""
def semantic_seed(request):
    from exp.r17_integration.common.request_rng import RequestRNG
    return RequestRNG(request).seed_for()


class RequestNoise:
    def __init__(self, seed, device="cpu"):
        import torch
        self.seed = int(seed)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

    @classmethod
    def from_context(cls, request, device="cpu"):
        return cls(semantic_seed(request), device)

    def fresh(self, device="cpu"):
        return RequestNoise(self.seed, device)

    def receipt(self):
        from .manifest import tensor_digest
        return {"seed": self.seed, "stream": "E_sequential_initial_then_posterior_v1",
                "state_digest": tensor_digest(self.generator.get_state())}
