"""Fail-closed R12 method selection shared by planning and serving."""
from __future__ import annotations

import json

METHODS = {
    'exp.offline_search.closed_loop.lookhalf:R12LookHalf': ('off', 'opus_m1', 'astra_consensus2', 'astra_strict'),
    'exp.offline_search.closed_loop.guard:R12Guard': ('off', 'h2', 'm2'),
}


def value(argv, flag):
    """Read one exact flag, rejecting duplicates and missing values."""
    found = []
    for i, item in enumerate(argv):
        if item == flag:
            if i + 1 == len(argv) or argv[i + 1].startswith('--'):
                raise ValueError(f'{flag} requires a value')
            found.append(argv[i + 1])
        elif item.startswith(flag + '='):
            found.append(item.split('=', 1)[1])
    if len(found) > 1:
        raise ValueError(f'duplicate {flag} refused')
    if found and not found[0]:
        raise ValueError(f'{flag} requires a value')
    return found[0] if found else None


def validate(argv, *, spec=None, kwargs=None):
    """Require the explicit flag and constructor to select the same R12 method."""
    selected = value(argv, '--os-lookhalf')
    spec = value(argv, '--os-method') if spec is None else spec
    if spec == 'exp.offline_search.closed_loop.half_everywhere:R12HalfEverywhere':
        try:
            from exp.offline_search.closed_loop.half_everywhere_config import validate_kwargs
        except ModuleNotFoundError:
            # Remote node.py can load this validator without the serving tree
            # on sys.path. Its companion is equally stdlib-only.
            import importlib.util
            from pathlib import Path
            module_spec = importlib.util.spec_from_file_location('half_configuration',
                Path(__file__).with_name('half_everywhere_config.py'))
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            validate_kwargs = module.validate_kwargs
        if selected != 'half_everywhere':
            raise ValueError('R12HalfEverywhere requires explicit --os-lookhalf half_everywhere')
        if kwargs is None:
            kwargs = json.loads(value(argv, '--os-kwargs') or '{}')
        validate_kwargs(kwargs, check_file=False)
        return selected
    if selected is None and spec not in METHODS:
        return None
    if spec not in METHODS:
        raise ValueError('--os-lookhalf requires an R12LookHalf or R12Guard method')
    if selected is None:
        raise ValueError(f'{spec} requires explicit --os-lookhalf METHOD')
    if selected not in METHODS[spec]:
        raise ValueError(f'--os-lookhalf invalid method {selected!r} for {spec}')
    if kwargs is None:
        kwargs = json.loads(value(argv, '--os-kwargs') or '{}')
    if not isinstance(kwargs, dict) or kwargs.get('method') != selected:
        raise ValueError('--os-lookhalf must match the explicit constructor method')
    return selected


def validate_row(row):
    """Validate emitted/planned arm selection without changing its tokens."""
    return validate(row.get('plugin_args', []), spec=row.get('method'), kwargs=row.get('kwargs') or {})


def validate_execution(extra, step):
    """Active R12 requires actual client acknowledgments, including step zero."""
    if (type(extra.get('executed_steps')) is not int or extra['executed_steps'] < 0
            or type(extra.get('exec_ok')) is not bool or type(extra.get('decision_id')) is not int
            or extra['decision_id'] != step):
        raise ValueError('R12_EXECUTION_AUDIT_REQUIRED: use the audited R12 worker; '
                         'executed_steps, exec_ok and sequential decision_id are required')


def validate_fitted(method, selected):
    """Refuse a mismatched pickle, including R12 classes imported via an alias."""
    family = getattr(method, 'family', None)
    if family == 'r12_half_everywhere' or selected == 'half_everywhere':
        if family != 'r12_half_everywhere' or selected != 'half_everywhere':
            raise ValueError('fitted half-everywhere method differs from explicit selection')
        from exp.offline_search.closed_loop.half_everywhere_config import validate_fitted as check
        check(method, dict(camera_table=method.camera_table, knob_method=method.knob_method, target=method.target))
        return
    if selected is not None or family in ('r12_lookhalf', 'r12_guard'):
        # Guards deliberately leave the inherited look-half method off.
        attribute = 'guard_method' if family == 'r12_guard' else 'lookhalf_method'
        if selected is None or getattr(method, attribute, None) != selected:
            raise ValueError('R12 configuration refused: fitted method differs from explicit --os-lookhalf')
