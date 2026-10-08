"""Sequential shadow decisions on fixed closed-loop B observations.

The observations/actions remain those of the full baseline. A propagated guard
trace is a diagnostic, not a changed-controller trajectory or SR estimate.
"""
from .dev_validation import *
from collections import Counter
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w

def half_query(table,fit,C,task,x,step):
    met=metric_from(table,'half',task);cr=table[f'half_{task}_rows']
    d=met.dist(x[None])[0];ix=np.argsort(d,kind='stable')[:16];rows=cr[ix]
    dk=d[ix].astype(np.float64)
    w=_kernel_w((dk-dk[0])[None],C['kref'])[0]
    w=(w/w.sum()).astype(np.float32)
    raw=np.tensordot(w,C['act'][rows],1)
    f=feat(x[None],raw[None],np.array([step]),np.array([task]),fit['sigma'])
    residual=predict(fit['heads'][task],f)[0].reshape(10,6)*fit['sigma'][:6]
    def corrected(scale):
        a=raw.copy();factor=np.float32(np.clip((2-float(d[ix[0]])/scale)/1.25,0,1))
        a[:,:6]+=np.float32(.5)*(residual*factor)
        return a
    return rows,w,float(d[ix[0]]),raw,corrected(fit['scale']),corrected(fit['loeo_scale'])

def prog_span(old_span,old_step,old_prog,step,prog,ep_len):
    return int(old_span+step-old_step) if old_step>=0 and (float(prog)-float(old_prog))*max(int(ep_len)-1,1)<=.5 else 0

def quant(x):
    x=np.asarray(x)
    return dict(n=len(x),mean=float(x.mean()),p50=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max())) if len(x) else dict(n=0)

def mean_or_none(x):return float(np.mean(x)) if len(x) else None

def bootstrap(A,values,mask=None,ratio=False):
    """Paired resampling of whole B initial-state episodes within task."""
    if mask is None:mask=np.ones(len(A['step']),bool)
    uids=np.unique(A['uid']); tasks=[];num=[];den=[]
    for u in uids:
        ix=(A['uid']==u)&mask; num.append(np.sum(values[ix]));den.append(ix.sum())
        tasks.append(int(A['task'][A['uid']==u][0]))
    num=np.array(num);den=np.array(den);tasks=np.array(tasks);rng=np.random.default_rng(121203)
    groups=[np.flatnonzero(tasks==t) for t in np.unique(tasks)]
    draws=np.concatenate([rng.choice(g,(2000,len(g)),replace=True) for g in groups],axis=1)
    if ratio:
        totals=den[draws].sum(1);samples=num[draws].sum(1)/np.maximum(totals,1)
        point=num.sum()/max(den.sum(),1)
    else:
        means=np.divide(num,den,out=np.full(len(num),np.nan),where=den>0)
        samples=np.nanmean(means[draws],1);point=float(np.nanmean(means))
    return dict(mean=float(point),ci95=np.quantile(samples,[.025,.975]).tolist(),replicates=len(draws))

def replay(A,C,lib,table,fit,decisions,protocol='propagated',config=AtlasConfig()):
    n=len(A['step']);half=np.zeros(n,bool);out={k:np.full(n,np.nan) for k in ('support','safe_mass','state_error')}
    rows=A['rows'].copy();weights=A['weights'].copy();raw=A['raw'].copy();corr=A['full'].copy();oldscale=corr.copy()
    spans=np.zeros(n,int);one_spans=np.zeros(n,int);distance=A['distance'].copy();reasons=[];firsts=[];events=[]
    last_uid=None;last_full=-1;previous=-1;span=0;first=None
    eligible=np.zeros(n,bool);life=np.zeros(n,bool);ref=np.full(n,-1,int)
    for i in range(n):
        uid=str(A['uid'][i]);s=int(A['step'][i]);t=int(A['task'][i]);seq=decisions[uid];d=seq[s]
        if uid!=last_uid:
            if last_uid is not None:firsts.append(dict(uid=last_uid,first=first))
            last_uid=uid;last_full=-1;previous=-1;span=0;first=None
        prevspan=0 if previous<0 else (span if protocol=='propagated' else int(A['span'][previous]))
        valid_tail=(s>=2 and not seq[s-2]['hit'] and seq[s-2].get('vision',True)
                    and seq[s-1].get('source',seq[s-1].get('src'))=='policy_tail'
                    and seq[s-1].get('extras',{}).get('policy_tail_lifecycle')==1
                    and seq[s-1].get('previous_executed_steps')==5 and d.get('previous_executed_steps')==5
                    and seq[s-1].get('exec_ok',False) and d.get('exec_ok',False))
        lifecycle=(A['look_reason'][i] not in (6,7,8)) or (A['look_reason'][i]==6 and valid_tail)
        life[i]=lifecycle
        reason=('warmup' if s<config.warmup else 'memory_age' if last_full<0 or s-int(A['step'][last_full])!=config.lag
                else 'pending_span' if prevspan>0 else 'periodic_due' if A['periodic'][i] else 'lifecycle' if not lifecycle else '')
        if reason:
            p=dict(half=False,reason=reason)
        else:
            eligible[i]=True;ref[i]=last_full
            p=plan(config,step=s,last_full_step=int(A['step'][last_full]),current_rs=A['rs'][i],last_full_rs=A['rs'][last_full],
                rows=A['rows'][last_full],weights=A['weights'][last_full],next2=table['next2'],library_rs=table['rs'],
                safe=table['safe'],state_scale=table['state_scale'],pending_no_progress=prevspan>0,
                periodic_due=bool(A['periodic'][i]),lifecycle_valid=lifecycle)
            for k in out:out[k][i]=p[k]
        half[i]=p['half'];reasons.append(p['reason'])
        if half[i]:
            # Current third-person projection is deliberately not an argument.
            x=half_features(last_full_v0=A['P'][last_full,:64],current_wrist=A['P'][i,64:],current_rs=A['rs'][i],last_full_rs=A['rs'][last_full])
            rows[i],weights[i],distance[i],raw[i],corr[i],oldscale[i]=half_query(table,fit,C,t,x,s)
        top=int(rows[i,0]);prog=C['progress'][top];lens=int(lib.ep_len[top])
        if previous<0:
            span=one_span=0
        else:
            oldtop=int(rows[previous,0]);span=prog_span(span,int(A['step'][previous]),C['progress'][oldtop],s,prog,lens)
            one_span=prog_span(int(A['span'][previous]),int(A['step'][previous]),A['progress'][previous],s,prog,lens)
        spans[i]=span;one_spans[i]=one_span
        call=span>=A['threshold'][i];basecall=A['span'][i]>=A['threshold'][i]
        if call!=basecall:
            events.append(dict(uid=uid,step=s,mode='half' if half[i] else 'full',kind='missed' if basecall else 'extra',
                base_span=int(A['span'][i]),shadow_span=span,full_top=int(A['rows'][i,0]),chosen_top=top))
            if first is None:first=dict(step=s,kind='missed_call' if basecall else 'extra_call',mode='half' if half[i] else 'full')
        if s+1<len(seq):
            # Under C10 a MISS consumes one policy-tail block; a HIT with
            # positive pending span requires a fresh look immediately.
            expected_next_vision=bool(span>0 and not call)
            base_next_vision=bool(seq[s+1].get('vision',True))
            if expected_next_vision!=base_next_vision and first is None:
                first=dict(step=s+1,kind='extra_look' if expected_next_vision else 'missed_look',mode='half' if half[i] else 'full')
        if not half[i]:last_full=i
        elif call or A['periodic'][i]:last_full=-1 # completion does not install half donors as full memory
        previous=i
    firsts.append(dict(uid=last_uid,first=first))
    return dict(half=half,rows=rows,weights=weights,raw=raw,corrected=corr,oldscale=oldscale,spans=spans,
        one_spans=one_spans,distance=distance,eligible=eligible,lifecycle=life,reference=ref,reasons=np.array(reasons),
        firsts=firsts,events=events,**out)

def summarize(A,R,C,meta,prices):
    half=R['half'];basecall=A['span']>=A['threshold'];call=R['spans']>=A['threshold']
    basecall=basecall|A['periodic'];call=call|A['periodic'];n=len(half);sig=C['sigma'][:6]
    delta=np.mean(((R['corrected'][:,:,:6]-A['full'][:,:,:6])/sig)**2,(1,2))
    rawdelta=np.mean(((R['raw'][:,:,:6]-A['raw'][:,:,:6])/sig)**2,(1,2))
    grip=np.mean((R['corrected'][:,:,6]>=0)!=(A['full'][:,:,6]>=0),1)
    top=R['rows'][:,0]==A['rows'][:,0]
    overlap=np.array([sum(min(float(w),float(ww[np.flatnonzero(rr==r)[0]])) for r,w in zip(r0,w0) if r in rr)
                      for r0,w0,rr,ww in zip(A['rows'],A['weights'],R['rows'],R['weights'])])
    target=A['shadow'];has=np.isfinite(target).all((1,2))
    def err(x):return np.mean(((x[:,:,:6]-target[:,:,:6])/sig)**2,(1,2))
    full_err=err(A['full']);candidate_err=err(R['corrected']);old_err=err(R['oldscale'])
    # CALL uses the available full-policy shadow sample. At baseline MISS it
    # is the real executed policy chunk, hence zero error by construction.
    available_policy=np.where(basecall[:,None,None],A['served'],target)
    proposed=np.where(call[:,None,None],available_policy,R['corrected']);served_err=err(proposed);base_served_err=err(A['served'])
    served_valid=np.isfinite(proposed).all((1,2))
    served_delta=np.mean(((proposed[:,:,:6]-A['served'][:,:,:6])/sig)**2,(1,2))
    served_grip=np.mean((proposed[:,:,6]>=0)!=(A['served'][:,:,6]>=0),1)
    full_grip=np.mean((A['full'][:,:,6]>=0)!=(target[:,:,6]>=0),1)
    candidate_grip=np.mean((R['corrected'][:,:,6]>=0)!=(target[:,:,6]>=0),1)
    missed=basecall&~call;extra=~basecall&call;localcall=R['one_spans']>=A['threshold']
    localmiss=basecall&~localcall&half;localextra=~basecall&localcall&half
    half_bools=(np.sqrt(rawdelta)<=.15)&(grip==0)
    firsts=R['firsts'];timing=[];miss_latency=[]
    for uid in np.unique(A['uid']):
        ix=np.flatnonzero(A['uid']==uid);b=ix[basecall[ix]];c=ix[call[ix]]
        timing.append(dict(uid=str(uid),baseline_first=int(A['step'][b[0]]) if len(b) else None,
            shadow_first=int(A['step'][c[0]]) if len(c) else None,
            delta=int(A['step'][c[0]]-A['step'][b[0]]) if len(b) and len(c) else None))
        for j in ix[missed[ix]]:
            after=ix[(A['step'][ix]>A['step'][j])&call[ix]]
            miss_latency.append(dict(uid=str(uid),step=int(A['step'][j]),next_shadow_call_delay_slots=int(A['step'][after[0]]-A['step'][j]) if len(after) else None))
    result=dict(looks=n,halves=int(half.sum()),half_share_pooled=float(half.mean()),half_share_episode_balanced=bootstrap(A,half.astype(float)),
        half_share_pooled_interval=bootstrap(A,half.astype(float),ratio=True),eligible_before_atlas=int(R['eligible'].sum()),
        reasons=dict(Counter(R['reasons'])),half_by_task={str(t):dict(half=int(half[A['task']==t].sum()),looks=int((A['task']==t).sum())) for t in range(10)},
        agreement=dict(top1_half=float(top[half].mean()),top1_all=float(top.mean()),weight_overlap_half=float(overlap[half].mean()),
            corrected_motion_rms_half=quant(np.sqrt(delta[half])),raw_motion_rms_half=quant(np.sqrt(rawdelta[half])),
            gripper_sign_rate_half=float(grip[half].mean()),any_gripper_disagreement_half=int((grip[half]>0).sum()),
            raw_action_safety_pass_half=float(half_bools[half].mean()),corrected_motion_mse_all_episode=bootstrap(A,delta)),
        served_agreement=dict(motion_rms_all=quant(np.sqrt(served_delta[served_valid])),motion_rms_half=quant(np.sqrt(served_delta[served_valid&half])),
            motion_mse_all_episode=bootstrap(A,served_delta,served_valid),gripper_sign_rate_all=float(served_grip[served_valid].mean()),
            any_gripper_disagreement_looks=int((served_grip[served_valid]>0).sum()),valid_looks=int(served_valid.sum()),unavailable_extra_policy_chunks=int((~served_valid).sum()),
            exact_full_chunk_equal_looks=int(np.all(proposed==A['served'],axis=(1,2)).sum()),
            note='Includes changed CALL/CACHE verdicts; counterfactual policy uses recorded full shadow sample.'),
        guards=dict(baseline_calls=int(basecall.sum()),shadow_calls=int(call.sum()),missed=int(missed.sum()),extra=int(extra.sum()),
            missed_on_half=int((missed&half).sum()),extra_on_half=int((extra&half).sum()),
            missed_on_later_full=int((missed&~half).sum()),extra_on_later_full=int((extra&~half).sum()),
            one_look_missed=int(localmiss.sum()),one_look_extra=int(localextra.sum()),
            episodes_missed=len(np.unique(A['uid'][missed])),episodes_extra=len(np.unique(A['uid'][extra])),
            first_divergence_counts=dict(Counter(x['first']['kind'] for x in firsts if x['first'])),
            prefixes_without_timing_divergence=sum(x['first'] is None for x in firsts),
            first_call_later=sum(x['delta'] is not None and x['delta']>0 for x in timing),
            first_call_earlier=sum(x['delta'] is not None and x['delta']<0 for x in timing),
            first_call_equal=sum(x['delta']==0 for x in timing),
            first_call_disappeared=sum(x['baseline_first'] is not None and x['shadow_first'] is None for x in timing),
            first_call_added=sum(x['baseline_first'] is None and x['shadow_first'] is not None for x in timing),
            first_call_both_absent=sum(x['baseline_first'] is None and x['shadow_first'] is None for x in timing),
            first_call_signed_delta_slots=quant([x['delta'] for x in timing if x['delta'] is not None]),
            missed_to_next_call_slots=quant([x['next_shadow_call_delay_slots'] for x in miss_latency if x['next_shadow_call_delay_slots'] is not None]),
            missed_with_no_later_shadow_call=sum(x['next_shadow_call_delay_slots'] is None for x in miss_latency)),
        guard_events=R['events'],first_divergences=firsts,first_call_timing=timing,missed_call_latency=miss_latency)
    if has.any():
        result['shadow_policy_error']=dict(
            full_proposal_all=bootstrap(A,full_err,has),mixed_proposal_all=bootstrap(A,candidate_err,has),
            proposal_delta_all=bootstrap(A,candidate_err-full_err,has),
            full_proposal_selected_half=bootstrap(A,full_err,has&half),half_proposal_selected_half=bootstrap(A,candidate_err,has&half),
            proposal_delta_selected_half=bootstrap(A,candidate_err-full_err,has&half),
            diagnostic_old_scale_proposal_all=bootstrap(A,old_err,has),
            baseline_served_all=bootstrap(A,base_served_err,has),shadow_served_all=bootstrap(A,served_err,has),
            served_delta_all=bootstrap(A,served_err-base_served_err,has),
            proposal_gripper_error_half_full=float(full_grip[has&half].mean()),proposal_gripper_error_half_half=float(candidate_grip[has&half].mean()),
            shadow_count=int(has.sum()),missed_calls_candidate_mse=mean_or_none(candidate_err[has&missed]),
            note='Episode-balanced standardized MSE over first ten controls/six motion channels; one stochastic shadow draw, not optimal-action truth. Counterfactual calls reuse this draw.')
    V,N,H,BC,CC=meta['V'],meta['N'],int(half.sum()),int((half&basecall).sum()),int((half&call).sum())
    stock=prices.get('stock_full',prices['full']);full=prices['full'];halfprice=prices['wrist_only'];completion=prices['completion_after_wrist'];policy=prices['policy_remainder']
    result['cost_diagnostic']=dict(prices=prices,baseline_calls_on_half=BC,shadow_calls_on_half=CC,
        saving_fixed_baseline_calls_stock=(stock-halfprice)*H/N-completion*BC/N,
        saving_fixed_baseline_calls_request=(stock-full)*V/N+(full-halfprice)*H/N-completion*BC/N,
        saving_fixed_schedule_shadow_calls_stock=(stock-halfprice)*H/N-completion*CC/N-policy*(call.sum()-basecall.sum())/N,
        saving_fixed_schedule_shadow_calls_request=(stock-full)*V/N+(full-halfprice)*H/N-completion*CC/N-policy*(call.sum()-basecall.sum())/N,
        break_even_half_share_no_completion=(full-stock)/(full-halfprice) if full>halfprice else None,
        note='Positive means IR saved vs stock full baseline. Observations/decision denominator/recorded look schedule held fixed; fewer guard calls are not an established saving or matched-SR benefit.')
    result['eligibility_gate_distributions']={k:quant(R[k][R['eligible']]) for k in ('safe_mass','support','state_error')}
    result['selected_gate_distributions']={k:quant(R[k][half]) for k in ('safe_mass','support','state_error')}
    result['agreement']['complete_raw_safety_label_pass_half']=float((half_bools[half]&(localcall[half]==basecall[half])).mean())
    result['agreement']['half_raw_grip_and_motion_safe_but_guard_disagrees']=int((half&half_bools&(localcall!=basecall)).sum())
    result['prefix_exposure']=dict(
        episodes=len(firsts),half_looks_before_or_at_first_timing_divergence=sum(int(half[(A['uid']==x['uid'])&
            (A['step']<=(x['first']['step'] if x['first'] else 10**9))].sum()) for x in firsts),
        looks_before_or_at_first_timing_divergence=sum(int(((A['uid']==x['uid'])&
            (A['step']<=(x['first']['step'] if x['first'] else 10**9))).sum()) for x in firsts))
    result['gate_rejections']=dict(support=int((R['eligible']&(R['support']<.99)).sum()),
        safety=int((R['eligible']&(R['safe_mass']<.75)).sum()),state=int((R['eligible']&(R['state_error']>.2)).sum()))
    return result

def assess(root,cell,size):
    start=time.monotonic();dest=DEST/root/f'{cell}_{size}';A=dict(np.load(dest/'looks.npz',allow_pickle=False))
    meta=json.loads((dest/'extract.json').read_text());decisions=json.loads((dest/'decisions.json').read_text())
    C,lib=load(cell,size);table=dict(np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz',allow_pickle=False));C['sigma']=table['sigma']
    with (DEST/'fits'/f'{cell}_{size}'/'half.pkl').open('rb') as f:fit=pickle.load(f)
    with Path(meta['fit_path']).open('rb') as f:recipe=pickle.load(f)['method']
    assert getattr(recipe,'knob_method','off')=='off', 'Use d2_screen for causal periodic replay'
    guard=recipe.inner; prev=-1;span=0;errors=[]
    for i in range(len(A['step'])):
        if i==0 or A['uid'][i]!=A['uid'][i-1]:
            prev=-1;span=0;guard.reset(NS(uid=str(A['uid'][i]),task_id=int(A['task'][i])))
        top=int(A['rows'][i,0]);s=int(A['step'][i]);prog=C['progress'][top]
        span=prog_span(span,int(A['step'][prev]) if prev>=0 else -1,C['progress'][int(A['rows'][prev,0])] if prev>=0 else 0,s,prog,lib.ep_len[top])
        deployed=guard._progress(NS(step=s,prev_hit=True),top)
        assert span==deployed==A['span'][i],(i,span,deployed,A['span'][i])
        assert bool(span>=A['threshold'][i])==bool(int(decisions[str(A['uid'][i])][s]['extras']['os_flags'])&8)
        assert bool((span>=A['threshold'][i]) or A['periodic'][i])==bool(not A['hit'][i])
        prev=i
    # DIRECTIVE_2: saturated centralized compute share, never request latency.
    vision=.152 if cell.startswith('pi05') else .148
    prices=dict(full=vision,stock_full=vision,wrist_only=vision/2,
                completion_after_wrist=vision/2,policy_remainder=1-vision,
                basis='DIRECTIVE_2 saturated compute share')
    result=dict(timestamp=now(),cell=cell,size=size,root=root,baseline=meta,full_guard_parity=len(A['step']),
        prediction_sha256=sha(HERE/'PREDICTION.md'),thresholds=vars(AtlasConfig()),calibrated_on_dev=False,protocols={})
    for protocol in ('propagated','anchored'):
        R=replay(A,C,lib,table,fit,decisions,protocol)
        result['protocols'][protocol]=summarize(A,R,C,meta,prices)
        np.savez_compressed(dest/f'{protocol}.npz',**{k:v for k,v in R.items() if isinstance(v,np.ndarray)})
        dump(dest/f'{protocol}_events.json',{k:R[k] for k in ('events','firsts')})
        r=result['protocols'][protocol]
        print(cell,protocol,'half',r['halves'],r['half_share_pooled'],'guard',r['guards']['missed'],r['guards']['extra'],flush=True)
    result['seconds']=time.monotonic()-start
    dump(dest/'assessment.json',result);dump(dest/'assess_reads.json',sorted(READS))
    return dict(cell=cell,size=size,seconds=result['seconds'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='r12_devlooks_50L');args=ap.parse_args()
    arms=json.loads((RUNS/args.root/'arms.json').read_text())
    for arm in arms:
        *_,cell,size=arm_info(args.root,arm)
        print(assess(args.root,cell,size),flush=True)
