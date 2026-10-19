"""CPU-only: which serving island reproduces the sealed DP model identity?"""
import json, sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
runtime, load_json = sys.argv[1], sys.argv[2]
sys.path[:0] = [runtime, runtime + '/src', runtime + '/packages/openpi-client/src']
from importlib.metadata import version
deps = {n: version(n) for n in ("torch", "torchvision", "diffusers", "transformers", "numpy", "safetensors")}
from exp.r17_integration.models.dp.config import LoadSpec
from exp.r17_integration.models.dp.loader import load_adapter
d = json.load(open(load_json))
fields = set(LoadSpec.__dataclass_fields__)
spec = {k: v for k, v in d.items() if k in fields}
expected = spec.get('expected_model_identity')
spec.update(device='cpu', expected_model_identity=None, diagnostic=True)  # identity digest excludes the diagnostic flag
a = load_adapter(spec)
actual = a.metadata().checkpoint_identity
print(json.dumps(dict(python=sys.executable, deps=deps, expected=expected, actual=actual, match=actual == expected)))
