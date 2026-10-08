"""Report frozen numerical gates separately from the owner's pilot permission."""
import json
from pathlib import Path
import numpy as np
from .build import HERE, STORE, write
from .encoder import sha


def phases(z):
    progress=np.load(STORE/'library/pi05_l10/bpool_cs/progress.npy',mmap_mode='r')
    p={key:progress[z[f'{key}_top'][:,0]] for key in ('student','reference')}
    counts={key:[0,0] for key in p}
    for ep in np.unique(z['episode']):
        ix=np.flatnonzero(z['episode']==ep)
        ix=ix[np.argsort(z['step'][ix],kind='stable')]
        for key in p:
            counts[key][0]+=int(np.count_nonzero(np.diff(p[key][ix])<-.1))
            counts[key][1]+=max(len(ix)-1,0)
    return dict(student_backward_gt01=counts['student'][0]/counts['student'][1],
        reference_backward_gt01=counts['reference'][0]/counts['reference'][1],
        discrepancy_gt_one_third=float(np.mean(np.abs(p['student']-p['reference'])>1/3)),
        absolute_phase_discrepancy_mean=float(np.mean(np.abs(p['student']-p['reference']))))


def gates():
    reports={}
    for size in (50,200):
        loeo=json.loads((HERE/f'evaluation/loeo_{size}.json').read_text())
        dev=json.loads((HERE/f'evaluation/dev_{size}.json').read_text())
        latency=json.loads((HERE/f'latency_{size}.json').read_text())
        lz=dict(np.load(loeo['data'],allow_pickle=False))
        dz=dict(np.load(STORE/f'derived/r15_qk/impl/pi05_l10_{size}/dev_refitted.npz',allow_pickle=False))
        lp,dp=phases(lz),phases(dz)
        accuracy=dict(library_balanced_ratio=loeo['excess']<=.05,
            library_all_tasks=max(t['excess'] for t in loeo['per_task'])<=.15,
            dev_balanced_ratio=dev['excess']<=.05,
            dev_bootstrap_upper=dev['excess_95_interval'][1]<=.10,
            dev_all_tasks=max(t['excess'] for t in dev['per_task'])<=.15,
            dev_top16_overlap=dev['top16_overlap']>=.75,
            dev_phase_backwards=dp['student_backward_gt01']-dp['reference_backward_gt01']<=.02,
            dev_phase_discrepancy=dp['discrepancy_gt_one_third']<=.05)
        reports[str(size)]=dict(proposal_accuracy_checks=accuracy,accuracy_pass=all(accuracy.values()),
            student_artifact_sha256=loeo['student_artifact_sha256'],
            representation_sha256=loeo['representation_sha256'],
            loeo_report_sha256=sha(HERE/f'evaluation/loeo_{size}.json'),
            dev_report_sha256=sha(HERE/f'evaluation/dev_{size}.json'),
            latency_report_sha256=sha(HERE/f'latency_{size}.json'),
            library_phase=lp,dev_phase=dp,
            loeo_rms10_balanced_ratio=1+loeo['excess'],dev_rms10_balanced_ratio=1+dev['excess'],
            dev_top16_overlap=dev['top16_overlap'],
            local_conservative_cost_pass=latency['graph_epsilon_measured']<=.0152 and latency['graph_epsilon_p95']<=.0304,
            deployment_host_cost_pass=None,
            frozen_epsilon=latency['frozen_artifact_epsilon'],measured_local_epsilon=latency['graph_epsilon_measured'],
            evidence='refitted guard/corrector proposals on old recorded trajectories, no closed-loop success claims')
    report=dict(cells=reports,offline_accuracy_pass=all(r['accuracy_pass'] for r in reports.values()),
        gates_path=str(HERE.parent/'astra/GATES.md'),gates_sha256=sha(HERE.parent/'astra/GATES.md'),
        unchanged_thresholds=True,owner_fourth_directive_allows_B_pilot=True,
        new_SR_measured=False,test_A_reads=0,closed_loop_launches=0,
        encoder_episode_cross_fitted=False,library_LOEO='donor exclusion with encoder fitted on full cell library')
    write(HERE/'gate_report.json',report)
    print('GATE_REPORT',report['offline_accuracy_pass'])
    return report


if __name__=='__main__':gates()
