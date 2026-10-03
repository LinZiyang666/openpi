"""R8 E2 discovery-only analysis. No serving code; captures remain read-only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader
from exp.offline_search.debug.tools.decision import common as C
from exp.offline_search.debug.tools.decision import call_value, stage_ledger, trigger_vs_onset
from exp.offline_search.debug.tools.decision.outcomes import endpoints
from exp.offline_search.debug.tools.physical import forensics
from exp.offline_search.debug.tools.physical.common import Episode

RUN = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E2_call_value')
PHYSICS = ['control_idx', 'decision_seq', 'is_settle', 'action', 'eef_pos', 'obj_pos', 'predicates']
CONFIG = dict(near_radius=.10, lift_height=.03, place_radius=.10,
              calibration_source='frozen published S6/E3 defaults; no callback outcome fitting')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(C.clean(value), indent=2, allow_nan=False) + '\n')


class DiscoveryArm:
    """Mask before any tool sees outcomes; disable capture-side derived caches."""
    def __init__(self, name):
        self.base = reader.open_arm(RUN, name)
        self.base.cache_enabled = False
        self.profile_metadata = {}
        self.profile_catalog_root = None
        self.eps = self.base.episodes()
        self.eps = self.eps[self.eps.init.astype(int).between(0, 29)].copy()
        self.ds = self.base.decisions()
        self.ds = self.ds[self.ds.episode_key.isin(self.eps.episode_key)].copy()
        assert len(self.eps) == 300 and set(self.eps.init.astype(int)) == set(range(30))
        assert self.ds.init.astype(int).between(0, 29).all()
        self._controls = {}
        self._catalog = None
        self.stage_column = None

    def __getattr__(self, key):
        return getattr(self.base, key)

    def episodes(self, accepted_only=True):
        assert accepted_only
        return self.eps.copy()

    def decisions(self, columns=None, accepted_only=True):
        assert accepted_only
        data = self.ds.copy()
        if self.stage_column:
            data['stage_pre'] = data[self.stage_column]
        return data if columns is None else data[columns]

    def controls(self, key, keys=None):
        assert key in set(self.eps.episode_key)
        if key not in self._controls:
            self._controls[key] = self.base.controls(key, PHYSICS)
        data = self._controls[key]
        return data if keys is None else {k: data[k] for k in keys}

    def catalog(self):
        if self._catalog is None:
            self._catalog = self.base.catalog()
        return self._catalog


def frozen_r7(arm):
    catalog = arm.catalog().set_index('row')
    result = []
    for r in arm.ds.to_dict('records'):
        rows, weights = r.get('rows'), r.get('weights')
        value = dict(r7_stage='unknown', r7_detail='unknown', event_mass=None)
        if isinstance(rows, list) and rows and isinstance(weights, list):
            c = catalog.loc[rows]
            w = np.asarray(weights, float)
            if len(w) == len(c) and np.isfinite(w).all() and w.sum() > 0:
                w = w / w.sum()
                event = float(w @ c.event_near.to_numpy(float))
                unknown = float(w @ (c['mode'].to_numpy(int) < 0))
                if unknown == 0:
                    mode = c['mode'].to_numpy(int)
                    if len(set(mode)) == 1 and mode[0] >= 0:
                        label = 'event' if event > 0 else 'interior'
                        runs = c.stage_run.to_numpy(int)
                        run = str(runs[0]) if len(set(runs)) == 1 else 'mixed'
                        value.update(r7_stage=label, r7_detail=f'run{run}_{label}_m{mode[0]}')
                    else:
                        value.update(r7_stage='mixed', r7_detail='mixed')
                value['event_mass'] = event
        result.append(value)
    for key in result[0]:
        arm.ds[key] = [r[key] for r in result]


def physical_labels(arm):
    labels, truth, proximal, prefix = [], {}, {}, []
    grouped = {key: group.to_dict('records') for key, group in arm.ds.groupby('episode_key', sort=False)}
    for meta in C.clean(arm.eps.to_dict('records')):
        meta['arm'] = arm.arm_name
        key = meta['episode_key']
        rows = C.clean(grouped[key])
        episode = Episode(meta, arm.controls(key), rows, dict(arm.server_meta, **arm.manifest), reader_arm=arm)
        try:
            label, data = forensics.analyse(episode, CONFIG)
            labels.append(label)
            carried = np.any([x['carried'] for x in data['relations']], axis=0) if data['relations'] else np.zeros(episode.n, bool)
            for r in rows:
                before = episode.before_index(r)
                did = r['decision_id']
                truth[did] = str(data['stage'][before]) if 0 <= before < episode.n else 'unknown'
                end = min(int(r['control_idx_start']) + 19, episode.n - 1)
                proximal[did] = float(carried[end]) - float(carried[before])
            # One independent prefix check per task, without fitting to outcomes.
            if int(meta['init']) == 0:
                cut = int(rows[len(rows)//2]['control_idx_start'])
                ep2 = Episode(dict(meta, n_controls=cut), {k: v[:cut] for k,v in episode.controls.items()},
                              [r for r in rows if int(r['control_idx_start']) < cut], episode.manifest)
                _, truncated = forensics.physical(ep2, CONFIG)
                prefix.append(dict(task_id=meta['task_id'], init=0, cut=cut,
                                   same=bool(np.array_equal(data['stage'][:cut], truncated['stage']))))
        except (ValueError, KeyError, IndexError) as exc:
            labels.append(dict(episode.identity, status='unavailable', reason=str(exc)))
            for r in rows:
                truth[r['decision_id']] = 'unknown'
                proximal[r['decision_id']] = None
    arm.ds['truth_pre'] = arm.ds.decision_id.map(truth)
    return labels, proximal, prefix


def one(name, bootstraps):
    start = time.monotonic()
    target = OUT / 'calls' / name
    print('start', name, flush=True)
    arm = DiscoveryArm(name)
    print('loaded', name, len(arm.ds), round(time.monotonic()-start,1), flush=True)
    frozen_r7(arm)
    ds, eps = C.inputs(arm)
    readiness = C.causal_readiness(ds, eps)
    assert not readiness, readiness
    labels, proximal, prefix = physical_labels(arm)
    print('physics', name, round(time.monotonic()-start,1), flush=True)
    save(target / 'onsets.json', labels)
    values = endpoints(arm, ds)
    for did, value in values.items():
        value['carry_gain_20'] = proximal[did]
    args = SimpleNamespace(bootstraps=bootstraps, seed=20261001, onsets=target/'onsets.json',
                           triggers=['os_c_stall_call', 'os_c3_dev_entry'])
    original_endpoints = call_value.endpoints
    reports = {}
    # Cache identical native endpoint computation across stage views; no tool file changes.
    call_value.endpoints = lambda a, d: values
    try:
        for stage_view in (None, 'r7_stage', 'truth_pre'):
            arm.stage_column = stage_view
            report = call_value.analyze(arm, args)
            report['stage_view'] = stage_view or 'stock_capture'
            report['stage_coverage'] = dict(arm.profile_stage_coverage)
            save(target / f'call_value_{stage_view or "stock"}.json', report)
            reports[stage_view or 'stock'] = {k:v for k,v in report.items() if k != 'tables'}
    finally:
        call_value.endpoints = original_endpoints
    arm.stage_column = 'r7_stage'
    ledger = stage_ledger.analyze(arm, args)
    save(target / 'stage_ledger.json', ledger)
    arm.stage_column = 'truth_pre'
    triggers = trigger_vs_onset.analyze(arm, args)
    save(target / 'trigger_vs_onset.json', triggers)
    records = []
    for r in arm.ds.to_dict('records'):
        did = r['decision_id']
        p, z, u = (C.number(r.get(k)) for k in ('p_effective', 'treatment', 'coin'))
        fresh = bool(r['vision']) and r['src'] not in ('cache_tail','cache_blind','policy_tail','follow') and r.get('chunk_offset') in (None, 0)
        records.append(dict(arm=name, episode_key=r['episode_key'], task_id=r['task_id'], init=r['init'],
            decision_id=did, decision_seq=r['decision_seq'], control_idx_start=r['control_idx_start'],
            n_applied=r['n_applied'], src=r['src'], vision=r['vision'], fresh=fresh,
            p=p, z=z, coin=u, p_nominal=r.get('p_nominal'), eligible=C.boolean(r.get('eligible')),
            call_count=r.get('policy_calls'), hit=r.get('hit'), owner_cost=r.get('owner_cost'),
            stage_pre_present=C.field(r,'stage_pre','pre_stage') is not None,
            r7_stage=r['r7_stage'], r7_detail=r['r7_detail'], event_mass=r['event_mass'], truth_pre=r['truth_pre'],
            dev_entry=C.number(C.field(r,'os_c3_dev_entry')), cooldown=C.number(C.field(r,'cooldown')),
            stall_state=C.number(C.field(r,'os_c_stall_state')), stall_call=C.number(C.field(r,'os_c_stall_call')),
            **{k:values[did][k] for k in ['final_success','remaining_work','remaining_active_controls','predicate_delta_5','predicate_delta_10','predicate_delta_20','carry_gain_20']}))
    frame = pd.DataFrame(records)
    frame.to_parquet(target/'decisions.parquet', index=False)
    eps[['episode_key','task_id','init','success','n_decisions','n_controls','termination_reason']].to_parquet(target/'episodes.parquet',index=False)
    anchors = frame[frame.fresh]
    supported = anchors[anchors.p.between(0,1,inclusive='neither') & anchors.eligible.eq(True)]
    summary = dict(arm=name, seconds=time.monotonic()-start, discovery_episodes=len(eps), decisions=len(frame),
        fresh=len(anchors), supported=len(supported), p_zero=int(anchors.p.eq(0).sum()), p_one=int(anchors.p.eq(1).sum()),
        coin_mismatches=int(((anchors.coin < anchors.p).astype(int) != anchors.z).sum()),
        call_count_mismatches=int((anchors.z != anchors.call_count).sum()),
        source_mismatches=int((anchors.z != (~anchors.hit.astype(bool)).astype(int)).sum()),
        stage_pre_present=int(frame.stage_pre_present.sum()),
        dev_entries=int(anchors.dev_entry.eq(1).sum()), dev_entries_supported=int(supported.dev_entry.eq(1).sum()),
        forensic_labels=dict(Counter(r.get('label',r.get('reason')) for r in labels)),
        physical_available=sum(r['status']=='available' for r in labels),
        prefix_checks=prefix, ledger=ledger['tables']['stages'][0], reports=reports,
        triggers=triggers['tables']['triggers'], reader_issues=arm.read_issues)
    save(target/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['arm','seconds','discovery_episodes','decisions','fresh','supported','p_zero','p_one','dev_entries','dev_entries_supported','physical_available','stage_pre_present','coin_mismatches']}),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['pi05','groot'])
    p.add_argument('--arms', nargs='+')
    p.add_argument('--bootstraps',type=int,default=1000)
    a = p.parse_args()
    specs = json.loads((RUN/'arms.json').read_text())
    names = a.arms or [r['arm'] for r in specs if r['r8']['variant'] in ('CU','CT','IP') and (not a.model or r['model']==a.model)]
    for name in names:
        if (OUT/'calls'/name/'summary.json').exists():
            print('already completed', name, flush=True)
            continue
        one(name,a.bootstraps)


if __name__ == '__main__':
    main()
