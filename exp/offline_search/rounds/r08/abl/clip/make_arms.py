"""Copy all eight exact A rows; only name, method and fit artifact change."""
import copy
import json
from pathlib import Path

from exp.offline_search.rounds.r06.p2_ablations.make_arms import sources, artifact

HERE = Path(__file__).resolve().parent
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r08_abl')
SPEC = 'exp.offline_search.rounds.r08.abl.clip.method:ClipAWM'


def a_references(model,suite,scale):
    short='sp' if suite=='spatial' else suite
    cell=f'{short}_{scale}'
    if model=='groot':
        stem,run=f'r5x_g_{cell}_tail1u','r05_x'
    elif cell=='sp_500':
        stem,run='r4b3_p_sp_500_tail1uc','r04_blind'
    else:
        stem,run=f'r5t_p_{cell}_tail1uc','r05_ptail'
    return [f'{run}:{stem}',*[f'r06_paper:{stem}_rep{i}' for i in (2,3)]]


def build():
    rows, provenance = [], []
    for (model,suite,scale,kind),(path,src) in sources().items():
        if kind != 'A':
            continue
        r = copy.deepcopy(src)
        r['name'] = f'r8abl_clip_{"p" if model=="pi05" else "g"}_{"sp" if suite=="spatial" else "l10"}_{scale}'
        r['method'] = SPEC
        r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1] = f'<RUN>/fits/{r["name"]}.pkl'
        rows.append(r)
        provenance.append({'arm':r['name'], 'source':str(path), 'source_artifact':artifact(src), 'source_row':src,
                           'A_references':a_references(model,suite,scale)})
    return rows, provenance


def main():
    rows, prov = build()
    (HERE/'arms_clip.json').write_text(json.dumps(rows,indent=1)+'\n')
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/provenance.json').write_text(json.dumps(prov,indent=1)+'\n')
    print('wrote 8 exact A-derived CLIP rows (name/method/fit only)')


if __name__ == '__main__':
    main()
