import exp.r17_integration.tests.models.groot.conftest as c
from exp.r17_integration.models.groot.manifest import GrootManifest, library_metadata, validate_library
ids = [c.digest(s) for s in ("weights", "norm", "code", "transforms", "tokenizer")]
m4, m2 = GrootManifest(*ids, denoising_steps=4), GrootManifest(*ids, denoising_steps=2)
print("served metadata has r17_model_spec:", "r17_model_spec" in m4.to_dict())
print("K4 vs K2 ModelSpec identity equal:", m4.model_spec().identity == m2.model_spec().identity)
try:
    validate_library(library_metadata(m4), m2)
    print("K4 teacher library accepted on K2 endpoint")
except ValueError as e:
    print("K4 teacher library on K2 endpoint ->", e)
m4.model_spec().layout.validate_commit(7)
print("commit=7 accepted by layout (replan5): returns wire[5:7] -> 2 rows < replan 5")
