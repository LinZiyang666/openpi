"""Four primary arm specs; separately identified phase-B2 variant and prefits."""
import shlex
from .common import HERE, ROOT, FITS, SPEC, PREFIX, arm_name, kwargs, write_json

def main():
    commands = []
    for variant in ('tail', 'phase2'):
        rows = []
        for suite in ('l10', 'spatial'):
            for scale in (50, 500):
                name = arm_name(suite, scale, variant)
                args = ['--os-root', str(ROOT), '--os-no-shadow-native', '--os-blind',
                        '--os-stage1-mode', 'wrist_only', '--os-tokens', 'off',
                        '--os-judge', 'guard_only', '--os-fit-artifact', f'<RUN>/fits/{name}.pkl']
                rows.append(dict(name=name, model='pi05', suite=suite, mode='plugin', method=SPEC,
                    kwargs=kwargs(scale, variant), full_model=True, cost_ledger=True,
                    plugin_args=args, yaml_patch={'miss': {'num_steps': 10,
                    'evidence_dir': f'<RUN>/evidence/{name}'}, 'write_policy': {'type': 'never'}}))
                # The plugin prefit entry point has no stage parser; stage is a server flag.
                fit_args = args[:]
                del fit_args[fit_args.index('--os-stage1-mode'):fit_args.index('--os-stage1-mode')+2]
                fit_args[-1] = str(FITS/(name+'.pkl'))
                commands.append(PREFIX+['-m', 'exp.offline_search.closed_loop.plugin', '--os-method', SPEC,
                    '--os-kwargs', __import__('json').dumps(kwargs(scale, variant), separators=(',', ':')),
                    '--os-cell', f'pi05_{suite}_cache', '--os-log-dir', str(HERE/'results/prefit'/name),
                    '--os-tag', name]+fit_args)
        write_json(HERE/('arms_q6.json' if variant == 'tail' else 'arms_phase2.json'), rows)
    write_json(HERE/'results/prefit_commands.json', commands)
    (HERE/'prefit.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n'+
        '\n'.join(shlex.join(c) for c in commands)+'\n')

if __name__ == '__main__':
    main()
