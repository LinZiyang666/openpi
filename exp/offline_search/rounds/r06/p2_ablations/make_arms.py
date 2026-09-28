"""Copy authoritative source spec fields; change only name, method, switch, artifact."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
RUN = RUNS / 'r06_abl'
STORE = '/home/weiland/trace_runs/offline_search_store'
MODULE = 'exp.offline_search.rounds.r06.p2_ablations'
PREFIX = ['taskset', '-c', '26-29,70-73', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src',
          '.venv/bin/python']


def sources():
    out = {}
    for model in ('pi05', 'groot'):
        for suite, short in (('l10', 'l10'), ('spatial', 'sp')):
            for scale in (50, 500):
                if model == 'groot':
                    p, name = RUNS / 'r05_x/arms_in.json', f'r5x_g_{short}_{scale}_tail1u'
                elif (suite, scale) == ('spatial', 500):
                    p, name = RUNS / 'r04_blind/arms_in.json', 'r4b3_p_sp_500_tail1uc'
                else:
                    p, name = RUNS / 'r05_ptail/arms_in.json', f'r5t_p_{short}_{scale}_tail1uc'
                rows = json.loads(p.read_text())
                row, = [r for r in rows if r['name'] == name]
                out[model, suite, scale, 'A'] = (p, row)
            if model == 'pi05':
                p, name = RUNS / 'r05_q1/arms_in.json', f'r5q1_c10_p_{short}_50'
            else:
                p, name = HERE.parent / 'p1_groot_commit/arms_p1.json', f'r6p1_c10_g_{short}_50'
            row, = [r for r in json.loads(p.read_text()) if r['name'] == name]
            out[model, suite, 50, 'B'] = (p, row)
    return out


def artifact(row):
    args = row['plugin_args']
    return args[args.index('--os-fit-artifact') + 1]


def main():
    metric, loo, provenance = [], [], []
    for (model, suite, scale, kind), (p, src) in sources().items():
        for guard in (['identity'] if kind == 'A' else ['stuck', 'terminal', 'overtime', 'no_progress']):
            row = copy.deepcopy(src)
            row['name'] = f"r6p2_{guard}_{'p' if model == 'pi05' else 'g'}_{'sp' if suite == 'spatial' else suite}_{scale}"
            if kind == 'A':
                row['method'] = MODULE + '.metric:IdentityBlindAWM'
            else:
                cls = 'TriggerCommitJudge' if model == 'pi05' else 'TriggerGrootCommitJudge'
                row['method'] = MODULE + '.judge:' + cls
                row['kwargs']['disabled_guard'] = guard
            args = row['plugin_args']
            args[args.index('--os-fit-artifact') + 1] = f"<RUN>/fits/{row['name']}.pkl"
            (metric if kind == 'A' else loo).append(row)
            provenance.append(dict(arm=row['name'], source=str(p), source_name=src['name'], kind=kind,
                                   source_artifact=artifact(src), source_row=src))
    assert len(metric) == 8 and len(loo) == 16
    for name, rows in [('arms_metric', metric), ('arms_trigger_loo', loo), ('results/provenance', provenance)]:
        (HERE / (name + '.json')).write_text(json.dumps(rows, indent=1) + '\n')
    print('wrote 8 metric + 16 trigger arms')


if __name__ == '__main__':
    main()
