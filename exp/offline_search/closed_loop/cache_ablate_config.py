"""Explicit pure-cache ablation selection; legacy flags keep their meaning."""
import json
import re
from .lookhalf_config import value

ARMS = ('both', 'wrist_sub', 'wrist_refit', 'agent_sub', 'agent_refit')
SPEC = 'exp.offline_search.closed_loop.cache_ablate:R12CacheAblate'


def validate(argv, *, spec=None, kwargs=None):
    selected = value(argv, '--os-cache-ablate')
    spec = value(argv, '--os-method') if spec is None else spec
    if selected is None and spec != SPEC:
        return None
    if spec != SPEC or selected not in ARMS:
        raise ValueError('R12CacheAblate requires explicit --os-cache-ablate ARM')
    kwargs = json.loads(value(argv, '--os-kwargs') or '{}') if kwargs is None else kwargs
    if not isinstance(kwargs, dict) or kwargs.get('method') != selected:
        raise ValueError('ablation flag must match constructor method')
    forbidden = ('--os-judge', '--os-lookhalf', '--os-policy-tail', '--os-rand', '--os-gpu',
                 '--os-stage1-mode', '--os-pack-prefix', '--os-record-look-shadow',
                 '--os-groot-path-check-every', '--os-oracle')
    if any(v.startswith(forbidden) for v in argv):
        raise ValueError('pure-cache ablation refuses judge, completion, policy, and alternative paths')
    if not all(v in argv for v in ('--os-blind', '--os-request-cameras', '--os-no-shadow-native')):
        raise ValueError('ablation requires blind and request-owned camera serving')
    if value(argv, '--os-tokens') != 'off' or not value(argv, '--os-fit-artifact'):
        raise ValueError('ablation requires frozen artifact and --os-tokens off')
    if not re.fullmatch('[0-9a-f]{64}', value(argv, '--os-cache-ablate-sha256') or ''):
        raise ValueError('ablation requires explicit frozen metric artifact SHA256')
    if set(kwargs) != {'library', 'episode_subset', 'method'}:
        raise ValueError('ablation has no knob/corrector options')
    return selected


def validate_row(row):
    return validate(row.get('plugin_args', []), spec=row.get('method'), kwargs=row.get('kwargs'))


def validate_fitted(method, selected):
    if selected is not None or getattr(method, 'family', None) == 'r12_cache_ablate':
        if selected is None or getattr(method, 'ablate_method', None) != selected:
            raise ValueError('fitted ablation differs from explicit camera/metric selection')
