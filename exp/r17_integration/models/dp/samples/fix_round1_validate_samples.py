"""CPU-only final pinned resident + bundle check; no server or GPU."""
import json
from pathlib import Path
import time

from exp.r17_integration.common.config import load_cache_config
from exp.r17_integration.common.provider import validate_binding
from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.provider import register_provider

base = Path('exp/r17_integration/models/dp/samples')
started = time.monotonic()
checks = []
for domain in ('libero', 'metaworld'):
    source = base / f'fix_round1_{domain}_cpu'
    adapter = load_adapter(json.loads((source / 'load_cpu.json').read_text()))
    provider = register_provider(adapter)
    assert adapter.strict_load == {'missing': [], 'unexpected': [], 'keys': 475}
    checks.append({'check': 'actual_pinned_loader', 'domain': domain, 'status': 'pass',
                   'model_identity': adapter.metadata().identity})
    for k in (1,2,4,10,100):
        config = load_cache_config(source / f'k{k}.yaml')
        bound, sampler = validate_binding(config.adapter_binding)
        assert bound is provider and sampler.native_timesteps[0] == 99
        assert sampler.family == ('ddpm' if k == 100 else 'ddim')
        if k == 1:
            assert sampler.config['quality_status'] == 'lossy'
        checks.append({'check': 'bound_yaml_actual_resident', 'domain': domain, 'K': k, 'status': 'pass'})
receipt = {'schema': 'r17.p4.bundle_load_receipt.v1', 'device': 'cpu', 'gpu_executed': False,
           'exit_code': 0, 'passed': len(checks), 'skipped': 0, 'checks': checks,
           'elapsed_seconds': time.monotonic()-started}
(base / 'fix_round1_loader_bundle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print({k:receipt[k] for k in ('passed','skipped','elapsed_seconds')})
