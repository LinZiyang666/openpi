"""Apply frozen offline gates to the primary CPU-trained candidate."""
import json
from datetime import datetime,timezone
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,dump

def main():
    analysis=json.loads((HERE/'analysis.json').read_text());result={'created_utc':datetime.now(timezone.utc).isoformat(),'method':'symmetric_cnn_action_cpu','online_authorized':False,'cells':{}}
    for size in (50,200):
        cell=f'pi05_l10_{size}';d=analysis[cell];m=d['methods'][result['method']];ref=d['methods']['deployed_key_geometry'];p=HERE/cell
        lo={x['name']:x for x in json.loads((p/'loeo_results.json').read_text())['methods']};tr=np.load(DERIVED/cell/'train.npz');lref=np.load(p/'loeo_deployed_key_geometry.npz')['err10'];lc=np.load(p/f"loeo_{result['method']}.npz")['err10']
        ratios={}
        for t in range(10):
            es=np.unique(tr['episode'][tr['task_id']==t]);a=np.mean([lc[tr['episode']==e].mean() for e in es]);b=np.mean([lref[tr['episode']==e].mean() for e in es]);ratios[str(t)]=float(a/b)
        lr=lo[result['method']]['episode_task_balanced_err10']/lo['deployed_key_geometry']['episode_task_balanced_err10']
        br=m['episode_task_balanced_err10']/ref['episode_task_balanced_err10'];btr={t:v['err10']/v['reference_err10'] for t,v in m['balanced_by_task'].items()}
        libchecks={'balanced_err10_ratio_le_1p05':lr<=1.05,'every_task_ratio_le_1p15':max(ratios.values())<=1.15}
        bchecks={'all_look_policy_labels':d['labelled_looks']==d['looks'],'balanced_err10_ratio_le_1p05':br<=1.05,'upper95_excess_le_0p10':m['balanced_relative_excess_ci95'][1]<=.10,'every_task_ratio_le_1p15':max(btr.values())<=1.15,'top16_overlap_ge_p75':m['top16_overlap']>=.75,'large_backwards_delta_le_p02':m['phase_large_backwards']-ref['phase_large_backwards']<=.02,'phase_wrong_third_le_p05':m['phase_wrong_third_vs_teacher']<=.05}
        latency=json.loads((HERE/f'latency_cpu_extra_{size}.json').read_text())
        lat=next(r for r in latency['results'] if r['name']==result['method'] and r['mode']=='torchscript' and r['cpu_threads']==1)
        costchecks={'local_median_le_1p026ms':lat['end_to_end']['median_ms']<=1.026,'local_p95_le_2p052ms':lat['end_to_end']['p95_ms']<=2.052,'timed_samples_ge_400':lat['end_to_end']['n']>=400}
        result['cells'][cell]={'library':{'ratio':lr,'per_task_ratios':ratios,'checks':libchecks,'pass':all(libchecks.values())},'b_dev':{'ratio':br,'upper95_excess':m['balanced_relative_excess_ci95'][1],'per_task_ratios':btr,'checks':bchecks,'pass':all(bchecks.values())},'local_cpu_cost':{'median_ms':lat['end_to_end']['median_ms'],'p95_ms':lat['end_to_end']['p95_ms'],'epsilon_graph':lat['epsilon_pi05_graph'],'checks':costchecks,'pass':all(costchecks.values())},'deployment_host_cost_verified':False,'full_five_layer_plugin_verified':False,'offline_pass':False}
    result['offline_pass']=all(c['offline_pass'] for c in result['cells'].values())
    dump(HERE/'gate_report.json',result)
    lines=['# Offline gate application', '', '**NO-GO for online work.** The primary CPU-trained candidate fails unseen B-dev accuracy. The prototype has not refitted or verified a complete five-layer plugin, and H100 deployment-host latency is unmeasured. No approval is implied by this report.', '', '| Cell | LOEO balanced ratio | Worst LOEO task ratio | B-dev balanced ratio | Worst B-dev task ratio | Top-16 pass | CPU cost pass locally |', '|---|---:|---:|---:|---:|---|---|']
    for cell,c in result['cells'].items():
        lines.append(f"| {cell} | {c['library']['ratio']:.4f} | {max(c['library']['per_task_ratios'].values()):.4f} | {c['b_dev']['ratio']:.4f} | {max(c['b_dev']['per_task_ratios'].values()):.4f} | {c['b_dev']['checks']['top16_overlap_ge_p75']} | {c['local_cpu_cost']['pass']} |")
    lines += ['', 'Exact booleans and ratios: `gate_report.json`; definitions: frozen `GATES.md`. Library LOEO conditions on the final trained representation. A library pass cannot substitute for unseen B-dev. Spatial lacks all-look labels; GR00T lacks exact library RGB, so neither is eligible.']
    (HERE/'OFFLINE_GATES.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
