"""Emit six paired B pilot arms; prove size-200 baseline reuse; no launches."""
import copy
import json
from pathlib import Path
import pickle
from exp.offline_search.closed_loop import devset
from exp.offline_search.rounds.r15.descent import prepare as descent
from .build import HERE, REPO, baseline, write
from .encoder import sha

TRACE = Path('/home/weiland/trace_runs/os_closed_loop')


def prepare():
    selection_path = REPO/'exp/offline_search/rounds/r15/descent/selection.json'
    selection = json.loads(selection_path.read_text())
    pairs = selection['model_suites']['pi05_l10']['pairs']
    assert len(pairs)==300 and len({tuple(p) for p in pairs})==300
    assert all(sum(p[0]==t for p in pairs)==30 for t in range(10))
    subset200 = devset.r10_subset('pi05','l10',200)
    assert all(i not in subset200['excluded_inits_by_task'][str(t)] for t,i in pairs)
    oldroot = TRACE/'r15_descent_pi05_l10_200'
    oldrows = {r['arm']:r for r in json.loads((oldroot/'arms.json').read_text())}
    oldmanifest = oldroot/'dev_selection.json'
    manifest = json.loads(oldmanifest.read_text())
    assert manifest['pairs']==pairs and manifest['seed']==selection['seed']==descent.SEED
    proofs=[]
    records=[]
    for size in (50,200):
        root=TRACE/f'r15_qk_pilot_pi05_l10_{size}'
        root.mkdir(parents=True,exist_ok=True)
        newmanifest=root/'dev_selection.json'
        newmanifest.write_bytes(oldmanifest.read_bytes())
        build_manifest=json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
        fits=build_manifest['records']
        specs,levels=[],[]
        for level in ('off','ir25'):
            newname=f'r15qk_pi05_l10_{size}_light_'+('off' if level=='off' else 'periodic_ir25')
            spec=descent.arm_spec(f'pi05_l10_{size}',newname,fits[level],newmanifest)
            spec['plugin_args'].append('--os-light-key')
            specs.append(spec);levels.append(level)
            path,old=baseline(size,level)
            fit=dict(path=str(path),sha256=sha(path),spec=old['spec'],kwargs=old['kwargs'])
            name=f'r15qk_pi05_l10_{size}_current_'+('off' if level=='off' else 'periodic_ir25')
            expected=descent.arm_spec(f'pi05_l10_{size}',name,fit,newmanifest)
            if size==50:
                specs.append(expected);levels.append(level)
            else:
                oldname='r15d_pi05_l10_200_'+('off' if level=='off' else 'periodic_ir25')
                actual=oldrows[oldname]
                assert actual['method']==expected['method'] and actual['kwargs']==expected['kwargs']
                assert actual['rollout_replicate_seed']==selection['seed'] and actual['knob_coin_seed']==0
                assert actual['client_overrides']==expected['client_overrides']
                assert actual['plugin_args']==expected['plugin_args']
                assert actual['recording'] is False and actual['full_model'] and actual['episodes']==300
                assert devset.sha(actual['yaml'])==actual['yaml_sha256']
                assert sha(path)==fits[level]['baseline_sha256']
                proofs.append(dict(level=level,reused_root=str(oldroot),reused_arm=oldname,
                    fit_path=str(path),fit_sha256=sha(path),manifest_sha256=sha(oldmanifest),
                    identical_pairs=True,seed=selection['seed'],knob_coin_seed=0,
                    identical_method_kwargs_plugin_args=True,identical_client_overrides=True,
                    yaml_sha256=actual['yaml_sha256'],full_policy_dose=10))
        write(root/'protocol.json',dict(dev=True,init_pool='B',model='pi05',suite='l10',size=size,
            episodes_per_arm=300,decision_logs=True,recording=False,seed=selection['seed'],
            selection_sha256=sha(selection_path),paired=True,training='cell library only',
            key_candidate=build_manifest['encoder_spec'].get('candidate_name','cnn_action_cpu'),matched_fixed_caps=True,
            IR_new='(epsilon*Q+M)/N; T=M',epsilon=fits['off']['kwargs']['encoder_spec']['epsilon'],
            online_authorization='OWNER_ADDENDUM fourth directive',evaluation_launched=False))
        rec=descent.emit(root,f'pi05_l10_{size}',specs,levels,pool='B',final_root=root)
        if size == 200:
            rows = json.loads((root/'arms.json').read_text())
            for row, proof in zip(rows, proofs, strict=True):
                old = oldrows[proof['reused_arm']]
                assert Path(row['yaml']).read_bytes() == Path(old['yaml']).read_bytes()
                proof.update(identical_server_template=True,
                             emitted_template_sha256=sha(row['yaml']),
                             identical_manifest_bytes=newmanifest.read_bytes()==oldmanifest.read_bytes())
        records.append(rec)
    write(HERE/'baseline_equivalence.json',dict(PASS=True,records=proofs,source_root=str(oldroot)))
    write(HERE/'pilot_roots.json',records)
    (HERE/'queue_pilot.txt').write_text(''.join(f'@h100 {r["root"]}|{arm}\n' for r in records for arm in r['arms']))
    print('READY',[(r['root'],r['arm_count']) for r in records])
    return records


if __name__=='__main__':
    prepare()
