"""Read-only import/version and transformers patch probe, without a GPU context."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    model = sys.argv[1]
    import torch
    import transformers
    import openpi
    import openpi_client
    if model == "pi05":
        from openpi.models_pytorch import pi0_pytorch
        from openpi.training import config
        imports = [pi0_pytorch.__file__, config.__file__]
    elif model == "groot":
        import gr00t
        from gr00t.model.policy import Gr00tPolicy
        imports = [gr00t.__file__, str(Gr00tPolicy)]
    else:
        imports = []
    packages = {}
    for name in ("torch", "transformers", "numpy", "jax", "safetensors", "flash-attn", "websockets", "tyro"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(json.dumps(dict(model=model, python=platform.python_version(), packages=packages,
                         torch_build=torch.__version__, torch_cuda=torch.version.cuda,
                         openpi=list(openpi.__path__), openpi_client=list(openpi_client.__path__), imports=imports), indent=1))
    patch = Path(transformers.__file__).parent / "models/siglip/modeling_siglip.py"
    if patch.exists():
        print("SIGLIP_SHA256 " + hashlib.sha256(patch.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
