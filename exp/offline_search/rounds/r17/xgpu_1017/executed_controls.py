"""CPU comparison of ACK-confirmed executed controls, including terminal tails."""
from collections import Counter,defaultdict
import numpy as np
from paired_analysis import KEY
from paired_identity import check as check_identity


def validate_ack(row, source):
    """Bind a terminal/nonterminal ACK to its original returned action chunk."""
    if row.get('schema')!='xgpu.client_execution.v1':
        raise ValueError('execution trace schema required')
    count=row.get('executed_count');ack=row.get('execution',{})
    if type(count) is not int or count<0 or count!=ack.get('executed_count'):
        raise ValueError('invalid execution count')
    if (row['request']!=source['request'] or row['wire_delivery']!=row['request'].get('decision_seq') or
            ack.get('decision_seq')!=row['wire_delivery'] or row['env_step']!=source['env_step']):
        raise ValueError('execution ACK clock/identity mismatch')
    full=np.asarray(source['actions'],float)
    part=np.asarray(row['actions'],float)
    if count==0 and row['actions']==[] and full.ndim==2:part=part.reshape(0,full.shape[1])
    if (full.ndim!=2 or not np.isfinite(full).all() or count>len(full) or part.shape!=(count,full.shape[1]) or
            not np.isfinite(part).all() or not np.array_equal(part,full[:count])):
        raise ValueError('executed actions do not match returned prefix')
    if type(row.get('terminal_ack')) is not bool:
        raise ValueError('explicit terminal ACK status required')
    return part


def expand(rows):
    episodes=defaultdict(dict);ends=defaultdict(list)
    for row in rows:
        key=tuple(row[k] for k in KEY);start=row['env_step'];count=row['executed_count']
        if type(start) is not int or start<0 or type(count) is not int or count<0:
            raise ValueError('invalid control clock')
        values=np.asarray(row['actions'],float)
        if count==0 and row['actions']==[]:values=values.reshape(0,row['action_width'])
        if values.ndim!=2 or len(values)!=count or not np.isfinite(values).all():
            raise ValueError('invalid executed action tensor')
        for i,value in enumerate(values):
            if start+i in episodes[key]:raise ValueError('duplicate executed control')
            episodes[key][start+i]=value
        if row.get('terminal_ack'):ends[key].append(start+count)
        if len(ends[key])>1:raise ValueError('duplicate terminal execution ACK')
    return episodes,ends


def survival(rows,label):
    events=Counter();censors=Counter();unknown=0
    for row in rows:
        if row['first_'+label+'_is_exact']:events[row['first_'+label]]+=1
        elif row[label+'_right_censored']:censors[row['last_common_control']]+=1
        else:unknown+=1
    risk=sum(events.values())+sum(censors.values());curve=[];remaining=risk;s=1.
    for step in sorted(set(events)|set(censors)):
        before=remaining
        s*=1-events[step]/before
        remaining-=events[step]+censors[step]
        curve.append(dict(control=step,at_risk=before,events=events[step],censored=censors[step],survival=s))
    quantiles={str(q):next((r['control'] for r in curve if 1-r['survival']>=q),None)for q in (.1,.5,.9)}
    return dict(curve=curve,quantiles=quantiles,known_episodes=risk,unknown_episodes=unknown,
        caution='descriptive Kaplan-Meier; unknown coverage is reported, not treated as censoring; no task-cluster CI')


def compare(left,right,action_atol=.001,population=None):
    bindings=check_identity(left,right)
    a,ae=expand(left);b,be=expand(right);result=[]
    episodes=set(a)|set(b)|set(ae)|set(be)
    if population is not None:
        planned={tuple(row[k]for k in KEY)for row in population}
        if len(planned)!=len(population) or not episodes<=planned:
            raise ValueError('execution traces differ from outcome population')
        episodes=planned
        if left:check_identity(population,left)
        if right:check_identity(population,right)
    for episode in sorted(episodes):
        x,y=a.get(episode,{}),b.get(episode,{});common=sorted(set(x)&set(y))
        row=dict(episode=list(episode),first_executed_action=None,first_executed_gripper=None,
            observed_controls=len(common),last_common_control=common[-1]if common else None,
            unmatched_controls=sorted(set(x)^set(y)),terminal_control_left=ae.get(episode,[]),
            terminal_control_right=be.get(episode,[]))
        for step in common:
            u,v=x[step],y[step]
            if u.shape!=v.shape or not u.size:raise ValueError('executed action layouts differ')
            if row['first_executed_action'] is None and np.max(np.abs(u-v))>action_atol:
                row['first_executed_action']=step
            if row['first_executed_gripper'] is None and np.sign(u[-1])!=np.sign(v[-1]):
                row['first_executed_gripper']=step
        for label in ('executed_action','executed_gripper'):
            first=row['first_'+label]
            row['first_'+label+'_is_exact']=first is not None and common[:common.index(first)+1]==list(range(first+1))
            # Right-censor at the shorter observed terminal horizon; a missing
            # nonterminal log tail never becomes evidence of no divergence.
            row[label+'_right_censored']=bool(common) and first is None and common==list(range(common[-1]+1)) and (
                common[-1]+1 in ae.get(episode,[])+be.get(episode,[]))
        result.append(row)
    return dict(schema='xgpu.executed_divergence.v1',episodes=result,action_atol=action_atol,scientific_bindings=bindings,
        population_scope='all supplied paired outcomes'if population is not None else 'trace-present episodes only; episodes missing on both sides cannot be counted',
        scope='zero-based controls after benchmark settling; ACK-confirmed wire-action prefixes selected for execution, before any benchmark controller/gripper transforms; no hidden-state inference',
        action_distribution=survival(result,'executed_action'),gripper_distribution=survival(result,'executed_gripper'))
