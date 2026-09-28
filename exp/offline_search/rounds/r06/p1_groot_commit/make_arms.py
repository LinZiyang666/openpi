"""Write arms_p1.json (GR00T B), arms_rep.json (A/B replicates, both models x 4 cells), prefit_commands.json.

Replicate rows copy the source rows of the existing run roots verbatim (method, kwargs, plugin_args incl. the
original absolute --os-fit-artifact, client_overrides, full_model, cost_ledger); only "name" changes. GR00T B
replicates copy this file's arms_p1 rows (with their <RUN> placeholder).
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
ROOT = '/home/weiland/trace_runs/offline_search_store'
METHOD = 'exp.offline_search.rounds.r06.p1_groot_commit.judge:GrootCommitJudge'
PREFIX = ('taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
          'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python')
FLAGS = ['--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only',
         '--os-no-shadow-native']
SUITE_SHORT = {'l10': 'l10', 'spatial': 'sp'}


def kwargs(scale):
    lib, kref = ('current', 5) if scale == 50 else ('big', 8)
    # Exactly C10's kwargs (r05_q1 r5q1_c10_p_*), i.e. the A base kwargs plus the guard / lifecycle settings.
    return {"base_kwargs": {"lib": lib, "kref": kref, "serving": "anchor_tail", "budget": 1, "gates": "budget_only"},
            "progress_guard": "noprog_span", "events": "none", "stuck_guard": "vision_confirmed",
            "policy_tail_gate": "lifecycle", "monitor": "off"}


def p1_rows():
    rows = []
    for suite in ('l10', 'spatial'):
        for scale in (50, 500):
            name = f'r6p1_c10_g_{SUITE_SHORT[suite]}_{scale}'
            rows.append({"name": name, "model": "groot", "suite": suite, "mode": "plugin", "method": METHOD,
                         "kwargs": kwargs(scale), "full_model": True, "cost_ledger": True,
                         "client_overrides": {"replan_steps": 5, "resize_size": 256},
                         "plugin_args": ["--os-root", ROOT, *FLAGS, "--os-fit-artifact", f"<RUN>/fits/{name}.pkl"]})
    return rows


def source_rows():
    """(model, kind, cell) -> existing spec row, read from the run roots' arms_in.json."""
    wanted = {
        ('pi05', 'A', 'l10_500'): ('r05_ptail', 'r5t_p_l10_500_tail1uc'),
        ('pi05', 'A', 'l10_50'): ('r05_ptail', 'r5t_p_l10_50_tail1uc'),
        ('pi05', 'A', 'sp_50'): ('r05_ptail', 'r5t_p_sp_50_tail1uc'),
        ('pi05', 'A', 'sp_500'): ('r04_blind', 'r4b3_p_sp_500_tail1uc'),
        ('pi05', 'B', 'l10_50'): ('r05_q1', 'r5q1_c10_p_l10_50'),
        ('pi05', 'B', 'l10_500'): ('r05_q1', 'r5q1_c10_p_l10_500'),
        ('pi05', 'B', 'sp_50'): ('r05_q1', 'r5q1_c10_p_sp_50'),
        ('pi05', 'B', 'sp_500'): ('r05_q1', 'r5q1_c10_p_sp_500'),
        ('groot', 'A', 'l10_500'): ('r05_x', 'r5x_g_l10_500_tail1u'),
        ('groot', 'A', 'l10_50'): ('r05_x', 'r5x_g_l10_50_tail1u'),
        ('groot', 'A', 'sp_500'): ('r05_x', 'r5x_g_sp_500_tail1u'),
        ('groot', 'A', 'sp_50'): ('r05_x', 'r5x_g_sp_50_tail1u'),
    }
    out = {}
    for key, (run, name) in wanted.items():
        rows = [r for r in json.loads((RUNS / run / 'arms_in.json').read_text()) if r.get('name') == name]
        assert len(rows) == 1, (run, name, len(rows))
        out[key] = dict(source=f'{run}/arms_in.json', row=rows[0])
    for row in p1_rows():
        cell = row['name'].split('_g_')[1]
        out[('groot', 'B', cell)] = dict(source='rounds/r06/p1_groot_commit/arms_p1.json', row=row)
    return out


def main():
    arms = p1_rows()
    (HERE / 'arms_p1.json').write_text(json.dumps(arms, indent=1) + '\n')
    src = source_rows()
    rep, provenance = [], []
    for model in ('pi05', 'groot'):
        for kind in ('A', 'B'):
            for cell in ('l10_50', 'l10_500', 'sp_50', 'sp_500'):
                entry = src[(model, kind, cell)]
                for r in (2, 3):
                    row = copy.deepcopy(entry['row'])
                    row['name'] = f"{entry['row']['name']}_rep{r}"
                    rep.append(row)
                    provenance.append(dict(name=row['name'], model=model, config=kind, cell=cell,
                                           copied_from=entry['source'] + ':' + entry['row']['name']))
    names = [r['name'] for r in rep]
    assert len(set(names)) == len(names) == 32
    (HERE / 'arms_rep.json').write_text(json.dumps(rep, indent=1) + '\n')
    (HERE / 'results' / 'arms_rep_provenance.json').write_text(json.dumps(provenance, indent=1) + '\n')
    commands = []
    for row in arms:
        cell = f"groot_{row['suite']}_cache"
        cmd = (f"{PREFIX} -m exp.offline_search.closed_loop.plugin --os-method {row['method']} "
               f"--os-kwargs '{json.dumps(row['kwargs'], separators=(',', ':'))}' --os-cell {cell} "
               f"--os-root {ROOT} --os-log-dir <RUN>/prefit_logs/{row['name']} --os-tag {row['name']} "
               f"{' '.join(FLAGS)} --os-fit-artifact <RUN>/fits/{row['name']}.pkl")
        commands.append(dict(arm=row['name'], cell=cell, command=cmd,
                             stdout=f"<RUN>/fits/{row['name']}.prefit.log"))
    (HERE / 'prefit_commands.json').write_text(json.dumps(commands, indent=1) + '\n')
    lines = ['#!/usr/bin/env bash', '# GR00T B prefits (RUN defaults to the r06 paper run root). Refuses existing artifacts.',
             'set -euo pipefail', 'cd /home/weiland/projects/openpi',
             'RUN=${RUN:-/home/weiland/trace_runs/os_closed_loop/r06_paper}', 'mkdir -p "$RUN/fits" "$RUN/prefit_logs"',
             'pids=()']
    for c in commands:
        cmd = c['command'].replace('<RUN>', '$RUN')
        lines.append(f"{cmd} > \"$RUN/fits/{c['arm']}.prefit.log\" 2>&1 & pids+=($!)")
    lines += ['for p in "${pids[@]}"; do wait "$p"; done', 'sha256sum "$RUN"/fits/r6p1_c10_g_*.pkl']
    (HERE / 'prefit.sh').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(p1=[r['name'] for r in arms], rep=len(rep)), indent=1))


if __name__ == '__main__':
    main()
