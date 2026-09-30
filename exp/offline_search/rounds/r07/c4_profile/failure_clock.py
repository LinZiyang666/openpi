"""Forensic decision/control clocks from P3 strict products; no intent inference."""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from . import common as C


def stall_map(paths):
    out={}
    for path in paths:
        with path.open() as f:
            for r in csv.DictReader(f):
                out[r.get('arm',''),r['uid'],int(r['step'])]=r.get('state',r.get('stall'))
    return out


def measured_control(r, grip_dim):
    action=C.decode(r.get('action_issued'))
    aperture=C.decode(r.get('after.observation_numeric.robot0_gripper_qpos'))
    eef=C.decode(r.get('after.observation_numeric.robot0_eef_pos'))
    contacts=C.decode(r.get('after.contacts'))
    objects={k:v for k,v in r.items() if k.startswith('after.observation_numeric.') and k.endswith('_pos') and 'robot' not in k and '_to_' not in k and v}
    return dict(control=int(r['control']),step=int(r['decision_step']) if r.get('decision_step') else None,
                wire_gripper=float(action[grip_dim]) if action and len(action)>grip_dim else None,
                aperture=float(np.linalg.norm(aperture)) if aperture is not None else None,
                eef=eef,contacts=len(contacts) if contacts is not None else None,object_positions=objects)


def audit_cell(cell,campaign,stalls,worst):
    base,lib,manifest=C.bank(cell);table=C.stage_table(lib,manifest,base)
    anchors={(r['arm'],r['uid'],int(r['step'])):r for r in C.tables(cell,campaign)}
    decisions=defaultdict(list);controls=defaultdict(list)
    for r in C.tables(cell,campaign,'decisions'):decisions[r['arm'],r['uid']].append(r)
    for r in C.tables(cell,campaign,'controls'):controls[r['arm'],r['uid']].append(measured_control(r,manifest['gripper_dim']))
    episodes=[];timeline=[];physical=[]
    for key,ds in sorted(decisions.items()):
        ds.sort(key=lambda r:int(r['step']));cs=sorted(controls[key],key=lambda r:r['control'])
        if not cs:raise ValueError('failure_clock requires strict controls, not server-only tables')
        starts={}
        for c in cs:
            if c['step'] is not None:starts.setdefault(c['step'],c['control'])
        active=None;first_dev=first_valve=first_stall=None
        episode_timeline=[];cursor=[]
        for r in ds:
            s=int(r['step']);a=anchors.get((*key,s))
            if a is not None:active=a
            if active is None:raise ValueError('decision before first anchor')
            rows=np.asarray(C.decode(active['retrieval.rows']),int);weights=np.asarray(C.decode(active['retrieval.weights']),float)
            with np.load(r['absolute_input_archive'],allow_pickle=False) as z:state=np.array(z['robot_state'],float)
            age=s-int(active['step'])
            delta,support=table.displacement(state,C.decode(active['state.normalized']),rows,weights,age)
            residual=table.deviation(state,rows,weights)
            task=int(r['task_id']);cut=table.deviation_p75.get(task)
            dev=bool(cut is not None and residual>cut);valve=bool(support and delta>table.valve_radius)
            status=stalls.get((*key,s),'not_replayed')
            control=starts.get(s)
            if dev and first_dev is None:first_dev=control
            if valve and first_valve is None:first_valve=control
            if status=='slow_confirmed' and first_stall is None:first_stall=control
            cursor.append(int(table.advance(rows,age)[0]))
            episode_timeline.append(dict(cell=cell,campaign=campaign,arm=key[0],uid=key[1],step=s,control=control,
                Y=int(r['Y']),stage=C.stage_label(table,rows,weights),anchor_step=int(active['step']),
                displacement=delta if support else None,state_residual=residual,deviation_p75=cut,
                deviation_crossing=dev,valve_crossing=valve,successor_supported=support,stall=status,
                confidence='recorded state and frozen-library proxy; not a physical failure label'))
        move=C.moves(np.asarray(cursor),np.arange(len(cursor))>0,lib['episode'],lib['step'],lib['next'])
        for r,code in zip(episode_timeline,move):r['reference_cursor_move']=C.FLAG[int(code)]
        timeline.extend(episode_timeline)
        commands=[c for c in cs if c['wire_gripper'] is not None]
        changes=[c['control'] for p,c in zip(commands,commands[1:]) if (p['wire_gripper']>=0)!=(c['wire_gripper']>=0)]
        eef=[c for c in cs if c['eef'] is not None]
        motions=[np.linalg.norm(np.asarray(a['eef'])-b['eef']) for a,b in zip(eef,eef[1:])]
        episodes.append(dict(cell=cell,campaign=campaign,arm=key[0],uid=key[1],task=int(ds[0]['task_id']),
            init=int(ds[0]['init']),Y=int(ds[0]['Y']),decisions=len(ds),controls=len(cs),
            terminal_failure_control=cs[-1]['control'] if int(ds[0]['Y'])==0 else None,
            first_state_deviation=first_dev,first_valve_alert=first_valve,first_confirmed_stall=first_stall,
            wire_command_transition_controls=changes,
            valve_to_terminal_controls=cs[-1]['control']-first_valve if first_valve is not None else None,
            aperture_missing=sum(c['aperture'] is None for c in cs),contacts_missing=sum(c['contacts'] is None for c in cs),
            object_pose_missing=sum(not c['object_positions'] for c in cs),eef_motion=C.quant(motions),
            physical_annotation='unannotated: contacts and object motion do not establish grasp/slip/release intent'))
        physical.extend(dict(cell=cell,campaign=campaign,arm=key[0],uid=key[1],**c) for c in cs)
    selected=sorted(episodes,key=lambda r:(r['Y'],r['first_valve_alert'] is None,-r['controls']))[:worst]
    return dict(cell=cell,campaign=campaign,episodes=episodes,worst=selected),timeline,physical


def main():
    p=C.parser(__doc__);p.add_argument('--campaigns',nargs='+',choices=['bval','p3'],default=['bval','p3'])
    p.add_argument('--stall-csv',nargs='*',type=C.Path,default=list(C.HERE.parents[1].joinpath('r06/ideation_Q3/stall').glob('pilot_status_*.csv')))
    p.add_argument('--worst',type=int,default=3);a=p.parse_args();stalls=stall_map(a.stall_csv)
    reports=[];timeline=[];physical=[]
    for cell in C.cells(a.cells):
        for campaign in a.campaigns:
            report,ts,cs=audit_cell(cell,campaign,stalls,a.worst);reports.append(report);timeline.extend(ts);physical.extend(cs)
            print(cell,campaign,'episodes',len(report['episodes']),'failed',sum(1-r['Y'] for r in report['episodes']),flush=True)
    C.write(a.out/'failure_clock.json',dict(schema='r7.c4.clock.v1',reports=reports,
        caveat='Descriptive chronology on original controller streams. State crossings are proxies. Terminal failure is timeout/outcome, not the onset of failure. Wire command polarity is not named close/release; exact model transform semantics would be needed for those labels. Privileged contacts/poses remain evaluation-only; intent unannotated. Actual active controls and warmup remain distinguishable.'))
    C.csv_write(a.out/'failure_decisions.csv',timeline);C.csv_write(a.out/'failure_controls.csv',physical)


if __name__=='__main__':main()
