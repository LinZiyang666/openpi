"""Additional descriptive checks and deterministic Chinese/technical report.

Run after mechanism_extract.py and mechanism_analyze.py, with their documented
CPU affinity and environment. All displayed numbers come from these scripts.
"""
from mechanism_analyze import *
from mechanism_extract import RUNS, REPO, INPUTS, rows, record

def exact_lengths(df,n):
    """Use actual client env-step counts, including a partial final chunk."""
    import ast
    source=record(REPO/'examples/libero/main.py')
    assignments=[x for x in ast.walk(ast.parse(source.read_text())) if isinstance(x,ast.AnnAssign)
                 and isinstance(x.target,ast.Name) and x.target.id=='num_steps_wait']
    assert len(assignments)==1
    wait=int(ast.literal_eval(assignments[0].value))
    lengths=[]
    for (root,arm),a in df.groupby(['root','arm'],sort=False):
        spec=next(r for r in read(RUNS/root/'arms.json') if r['arm']==arm)
        warmup=int(spec.get('client_overrides',{}).get('num_steps_wait',wait))
        episodes=a.groupby('ep').first();seen=set()
        for r in rows(RUNS/root/'runs'/arm/'client/per_step.jsonl'):
            if r.get('_kind')!='client_timing' or not r.get('accepted'):continue
            task,init=map(int,r['task_uid'].split(':')[-2:]);ep=f'{root}/{arm}/{task}:{init}'
            if ep not in episodes.index or r['attempt']!=episodes.loc[ep,'attempt']:continue
            assert ep not in seen;seen.add(ep)
            N=int(episodes.loc[ep,'N']);actual=int(r['steps'])-warmup
            assert int(r['infers'])==N and 5*(N-1)<actual<=5*N,(ep,N,actual)
            lengths.append(dict(ep=ep,N=N,final_controls=actual,warmup_controls=warmup,unused_final_chunk_controls=5*N-actual))
        assert seen==set(episodes.index)
    by={r['ep']:r['final_controls'] for r in lengths}
    df['final_controls']=df.ep.map(by)
    df['fraction']=df.controls/df.final_controls
    n['timing_audit']=dict(episodes=len(lengths),partial_final_chunks=sum(r['unused_final_chunk_controls']>0 for r in lengths),
        unused_controls=dist([r['unused_final_chunk_controls'] for r in lengths]),
        final_length_source='accepted client_timing.steps minus configured warm-up; infers and final partial-chunk bounds checked for every episode.')
    n['definitions']['final_fraction']='elapsed controls / actual final controls from accepted client_timing, excluding warm-up and including partial final chunks; retrospective, not an online feature.'
    dump(HERE/'mechanism_episode_lengths.json',dict(episodes=lengths,inputs=dict(INPUTS),audit=n['timing_audit']))
    n['episode_lengths_sha256']=sha(HERE/'mechanism_episode_lengths.json')
    return df

def supplemental(df,n):
    cfgs={(c['cell'],c['method'],float(c['target'])):c for c in read(HERE/'configs.json')}
    for r in n['arms']:
        a=df[df.arm==r['arm']]
        r['placement']=placement(a)
        r['position_hazard']={c:position_hazard(a,c) for c in ['controls','fraction','progress']}
    for r in n['pooled_placement']:
        target_mask=df.target.isna() if r['target'] is None else df.target==r['target']
        a=df[(df.source==r['source'])&(df.model==r['model'])&(df.method==r['method'])&target_mask]
        r['placement']=placement(a)
    primary=df[(df.source=='pilot')&df.method.isin(['periodic','distance_feedback'])]
    n['matched_init_positions']=matched_positions(df)
    n['constant_score_replay']=[];n['same_state_nearest']=[]
    for (cell,target,method,rep),a in primary.groupby(['cell','target','method','rep']):
        cv,cm=COSTS[a.model.iloc[0]];cfg=cfgs[(cell,'distance_feedback',float(target))]
        alternative=[];natural=[];guard=[];near=[];runs=[]
        for _,e in a.groupby('ep',sort=False):
            V=M=0;run=0;maxrun=0
            for r in e.itertuples():
                theta=cfg['bias']+(cv*V+cm*M-target*r.step)/cm/cfg['width']
                alt=bool(r.guard or not np.isfinite(r.distance) or .5>theta)
                alternative.append(alt);natural.append(r.replay_d);guard.append(r.guard)
                V+=1;M+=alt
                run=0 if r.call else run+1;maxrun=max(maxrun,run)
            runs.append(maxrun)
            d=e[(~e.guard)&e.replay_d].controls.to_numpy();p=e[(~e.guard)&e.replay_p].controls.to_numpy()
            if len(d) and len(p):near.extend(np.r_[np.abs(d[:,None]-p).min(1),np.abs(p[:,None]-d).min(1)])
        natural=np.array(natural);alternative=np.array(alternative);eligible=~np.array(guard)
        meta=dict(cell=cell,target=float(target),behavior=method,rep=int(rep))
        n['constant_score_replay'].append(dict(meta,knob=agreement(natural[eligible],alternative[eligible]),
                                               all_calls=agreement(natural,alternative)))
        n['same_state_nearest'].append(dict(meta,nearest_controls=dist(near),within_10_controls=rate(np.asarray(near)<=10),
            maximum_uninterrupted_cache_looks_per_episode=dist(runs)))
    labels=primary[(primary.target==.32)&(primary.rep==0)&np.isfinite(primary.risk10)].copy()
    labels['log_risk10']=np.log1p(labels.risk10)
    # Recompute after tightening clusters to pair physical initializations
    # across models as well as the two behavior trajectories.
    n['conditional_shadow']=dict(pooled={m:conditional_shadow(labels,m) for m in ['risk5','risk10','grip10','log_risk10']},
        by_cell=[dict(cell=c,**conditional_shadow(a)) for c,a in labels.groupby('cell')],
        by_behavior=[dict(behavior=m,**conditional_shadow(a)) for m,a in labels.groupby('method')])
    n['shadow_selected_means']=[]
    for (cell,method),a in labels[~labels.guard].groupby(['cell','method']):
        for name,q in [('distance_all',a.replay_d),('periodic_all',a.replay_p),('distance_only',a.replay_d&~a.replay_p),
                       ('periodic_only',a.replay_p&~a.replay_d),('both',a.replay_d&a.replay_p),('neither',~a.replay_d&~a.replay_p)]:
            z=a[q]
            n['shadow_selected_means'].append(dict(cell=cell,behavior=method,selection=name,n=len(z),
                risk5_mean=float(z.risk5.mean()),risk10_mean=float(z.risk10.mean()),grip10_mean=float(z.grip10.mean())))
    d=primary[(primary.method=='distance_feedback')&(primary.target==.32)&(primary.rep==0)]
    z=primary[(primary.target==.32)&(primary.rep==0)]
    e=z[~z.guard]
    n['headline']=dict(replay32=agreement(e.replay_d,e.replay_p),replay32_all=agreement(z.replay_d,z.replay_p),
        feedback32=placement(d),distance_growth_raw_R2_range=[min(r['raw_within_episode_gap_R2'] for r in n['distance_growth'] if r['source']=='pilot' and r['method']=='distance_feedback'),
            max(r['raw_within_episode_gap_R2'] for r in n['distance_growth'] if r['source']=='pilot' and r['method']=='distance_feedback')])
    n['definitions']['conditional_shadow']='Cell x behavior x task x actual gap fixed effects; common-support harmonic weights; 1000 paired physical-init clusters within suite/task, jointly across both models and both behavior paths.'
    n['scripts'].update({p.name:sha(p) for p in [HERE/'mechanism_analyze.py',HERE/'mechanism_report.py',HERE/'mechanism_validate.py']})
    return n

def fmt(x,k=3):return '—' if x is None or not np.isfinite(x) else f'{x:.{k}f}'
def pct(x,k=1):return '—' if x is None or not np.isfinite(x) else f'{100*x:.{k}f}%'
def quant(d):return '/'.join(fmt(d.get('p'+p),1) for p in ['10','50','90']) if d.get('n') else '—'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def ci(r):return f"{fmt(r['distance_minus_periodic'],4)} [{fmt(r['bootstrap_95ci'][0],4)}, {fmt(r['bootstrap_95ci'][1],4)}]"

def render(n,df):
    h=n['headline'];s=n['conditional_shadow']['pooled']['risk10'];p=n['predictors'];g=h['feedback32']
    short=g['knob_shorter_than_periodic_min']/g['knob_postfirst_count']
    lines=[
        '# 策略调用位置：机制分析',
        '',
        f"距离触发并不等同于定期触发。两者的平均节奏相近，中等预算下都常在隔两次观察后调用；但把两条规则放到同一批已记录状态上，排除保护调用后，有 {pct(h['replay32']['symdiff_over_union'])} 的被选位置只被其中一条规则选中。距离规则更像是在预算约束的节奏内，根据当前状态提前或推迟调用。这个数字指具体观察时刻，并不表示它们在整段任务里相隔很远。",
        '',
        f"“等得越久、离示范越远”不足以解释数据。扣除每条轨迹自身的距离水平后，间隔只解释 {pct(h['distance_growth_raw_R2_range'][0])}–{pct(h['distance_growth_raw_R2_range'][1])} 的原始距离变化。调用后距离也不保证下降。中等预算下，距离规则有 {pct(short)} 的非首次主动调用早于对应定期规则允许的最短间隔，同时还留下了 {g['cache_looks_beyond_periodic_max']:,} 次定期规则按最长间隔也该打断的缓存观察。",
        '',
        f"你的直觉有一部分是对的：调用节奏很重要。固定模型、库和预算后，只用间隔预测距离规则是否调用，留出初始状态验证的准确率为 {pct(p['models']['gap']['accuracy'])}。不过这里还有一个会随等待而变化的量——累计预算账本。加入它之后，再加入扣除了间隔影响的距离，预测准确率从 {pct(p['models']['gap_ledger']['accuracy'])} 提高到 {pct(p['models']['gap_ledger_distance_residual']['accuracy'])}。因此，节奏主要来自预算约束；距离在这个节奏里参与选点。",
        '',
        f"选点差异有一些局部价值证据。在有影子动作标签的四组设置里，控制任务、轨迹来源和相同间隔后，距离规则独自选择的位置，其策略与修正缓存动作误差平均比定期规则独自选择的位置高 {pct(s['ratio']-1)}。差异主要来自空间任务；一组长任务设置方向相反。因此可以说它在部分设置里把调用移向了更可能与缓存动作不一致的位置，尚不能说这种移动提高了任务成功率。",
        '',
        '这里只用了开发验证记录，没有读取最终测试结果，也没有修改任何控制参数、阈值、预算或实验单元。',
        '',
        '---',
        '',
        '**技术口径与数据边界**',
        '',
        f"`mechanism_extract.py → mechanism_analyze.py → mechanism_report.py` 生成本文及 `mechanism_numbers.json`。数据：{n['counts']['pilot_arms']} 个 B pilot arm / {n['counts']['pilot_episodes']:,} episodes；另有 4 个完整 devrec arm / {n['counts']['episodes']-n['counts']['pilot_episodes']} episodes。总计 {n['counts']['looks']:,} real looks。所有原始数据读取限定为六个 `r13_pilot_*` 根和两个 `r13_devrec_*` 根。旧 distance 臂尚无完成记录，不能报告其调用位置；没有用其他历史数据补齐。",
        '',
        'D = distance_feedback；P = periodic；S = distance_static_b（诊断臂）；O = off（只有保护调用）。MISS = 任意真实策略调用；knob = 非保护规则调用。每 decision slot 执行 5 controls；gap 是自上一次任意 MISS 后的真实观察数，紧接下一次观察记 1。首次调用没有伪造的“上一次调用”，单列 startup。knob→knob 间隔允许中间夹有保护调用。终止时未完成的间隔按删失记录保存在 JSON，不当作完整调用间隔。',
        '',
        'P 使用冻结 cap：`floor(setting) + 1[U(task, init, last_call_step) < frac(setting)]`；当前 look 之前的连续 cache-look 数达到 cap 才调用，所以无保护时的调用间隔是 cap+1。回放使用原始 SHA256 keyed coins 和各自的 last_call_step。D 使用冻结 `u = clip((log1p(raw_distance)-lo)/(hi-lo),0,1)`，`theta = bias + (cv*V + cm*M - target*N)/(cm*width)`，按 `u > theta` 调用；每条反事实规则维护自己的调用历史/账本。',
        '',
        'raw_distance 来自每次真实观察的几何距离；绝不把记录的 kernel scores 或 `-scores` 当作几何距离。devrec 未直接记录 raw_distance，由冻结模型和已记录 key/state 在 CPU 上重建。距离没有显式的“距上次调用时间”项（初始状态有冻结的专用几何分支）；MISS 也不把机器人位置强制投到示范轨迹上。继续执行会累积目标预算额度；不调用时超支量下降、theta 随之降低，即使距离不变也可能触发下一次调用。',
        '',
        '相同状态回放固定实际状态、look/guard/fallback 时刻；不重算反事实动作之后的状态、保护机制或观察时序。它是选点比较，不能给出反事实成功率。真实双轨迹只在配对 init 上比较分布，不视为逐状态对应。episode fraction = 已执行 controls / 该轨迹实际总 controls；总长使用 client_timing.steps 扣除 warm-up，保留末段不足 5 步的提前终止。它是回顾性位置，包含终止选择效应。`top1_prog` 是检索到的示范归一化进度，切换 donor 时会跳变。',
        '',
        '各表是描述性结果。重复块不冒充新 init；shadow 区间按 suite/task 内的物理 init 重采样，并把两个模型、两种行为路径一起抽取。预测器按 cell/task/init 划五折，所有 dose/repeat 留在同一折。合并表按 look/call 加权，不代表等 cell 权重的性能估计。',
        '',
        '主机：groot 全部 pilot 与 pi05 L10-50 为 `weilandserver`（4090）；pi05 L10-200 / Spatial-50 为 `vla-cache`（H100）。遵循协调裁定 D2；不据此重新放行或否决实验。D4 后续 test-A 只分析 rep0/rep1；本文没有读取 test-A 数据，也没有改动冻结方案。',
        '',
        '**调用位置：每个 cell / dose / repeat**',
        '',
        '下表 gap 分位数单位为真实观察（p10 / p50 / p90）。`K←M` 指 knob 调用距上次任意 MISS，`K←K` 指 knob 调用距上次 knob。位置列为 knob 调用的中位数；JSON 同时含任意调用、保护调用的完整分布、controls 间隔和按位置的 hazard。',
        ''
    ]
    rows=[]
    names={'distance_feedback':'D','periodic':'P','distance_static_b':'S','off':'O'}
    for r in sorted([r for r in n['arms'] if r['source']=='pilot'],key=lambda r:(r['cell'],r['target'],r['rep'],r['method'])):
        z=r['placement'];pos=z['positions']['knob']
        rows.append([r['cell'],fmt(r['target'],2),r['rep'],names[r['method']],z['calls'],z['knobs'],pct(z['guard_share_of_calls']),
            quant(z['gap_any_call']),quant(z['gap_knob_since_any']),quant(z['gap_knob_since_knob']),
            fmt(pos['controls'].get('p50'),0),fmt(pos['fraction'].get('p50'),2),fmt(pos['progress'].get('p50'),2)])
    lines += [table(['Cell','Dose','Rep','Rule','MISS','Knob','Guard/MISS','M←M','K←M','K←K','Elapsed controls','Episode fraction','Demo progress'],rows),'',
        '**非保护调用 hazard 与保护机制**','',
        'Hazard 分母为该 gap 的非保护真实观察。≥5 将所有更长间隔合并；首次调用前的观察另算，未混入这些 gap。短间隔比例只计已有前次 MISS 的 knob。长缓存数表示 gap 已达到周期规则最大允许间隔而 D 仍不调用的 eligible looks；这是计数，不是新增独立 episode。','']
    rows=[]
    for (model,target,method),a in df[(df.source=='pilot')&df.method.isin(['periodic','distance_feedback'])].groupby(['model','target','method']):
        z=placement(a);e=a[~a.guard&~a.startup]
        hs=[rate(e[e.gap==gap].knob) for gap in (1,2,3,4)]+[rate(e[e.gap>=5].knob)]
        rows.append([model,fmt(target,2),names[method],*[pct(v) for v in hs],pct(z['knob_after_guard_fraction']),
            pct(z['knob_at_first_look_after_guard_fraction']),
            pct(z['knob_shorter_than_periodic_min']/max(z['knob_postfirst_count'],1)),z['cache_looks_beyond_periodic_max']])
    lines += [table(['Model','Dose','Rule','g=1','g=2','g=3','g=4','g≥5','Prev MISS guard / K','First look after guard / K','K below P min','Cache at/beyond P max'],rows),'',
        '**距离是否只是等待时间**','',
        '下表回归在每条 episode 内去均值，再用 gap 分类项解释 raw_distance；它是样本内的共同间隔曲线 R²，不是因果效应。相关列为逐 episode Spearman 的中位数。距离升高会触发调用并截断长 gap，因此幸存到长 gap 的状态有选择偏差；负相关不能解释为“调用使距离增大”。同时展示 P 和 off 的路径，避免只看距离规则自身的选择结果。','']
    rows=[]
    for r in n['distance_growth']:
        if r['method'] not in ['periodic','distance_feedback','off']:continue
        rows.append([r['source'],r['cell'],names[r['method']],pct(r['raw_within_episode_gap_R2'],2),pct(r['log1p_within_episode_gap_R2'],2),
            fmt(r['episode_spearman'].get('p50')),pct(r['uninterrupted_run_fraction_distance_increases']),pct(r['next_look_after_policy_fraction_distance_decreases'])])
    lines += [table(['Source','Cell','Rule','Raw within-ep R²','Log within-ep R²','Median ep ρ','Distance rises within cache run','Distance falls next look after MISS'],rows),'',
        f"跨初始状态验证中，按 cell/dose/gap 预测 log-distance，相对 cell/dose 均值基线的 R² 为 {fmt(p['heldout_log_distance_gap_R2_vs_cell_dose_mean'])}；它包含 episode 间组成变化，不能与上表 within-episode R² 混为一谈。调用决定的 gap-only 预测器含 cell/dose×gap 截距；distance residual 在训练折中扣除对应 cell/dose/gap 均值。ledger 是决策前实际 theta。没有把 episode 身份或最终长度作为预测特征。",
        '',
        table(['Predictor','Held-out looks','AUC','Brier','Log loss','Accuracy'],[[m,v['n'],fmt(v['auc']),fmt(v['brier']),fmt(v['logloss']),pct(v['accuracy'])] for m,v in p['models'].items()]),'',
        f"只控制 cell/dose/gap 的距离残差与调用决定的偏相关为 {fmt(p['partial_distance_call_correlation_given_cell_dose_gap'])}：仅加距离的总体改善较小。预算账本解释了更多；在相同账本背景下，距离仍提供清楚的额外区分。最后一行接近重建规则本身，不能用作效益证明。下面单列 .32，避免合并 dose 的先验调用率差异夸大 AUC。",'']
    rows=[]
    for r in p['by_cell_dose']:
        if r['target']==.32:rows.append([r['cell'],*[fmt(r['models'][m]['auc']) for m in p['models']]])
    lines += [table(['Cell (.32)','Gap AUC','Gap+distance AUC','Gap+ledger AUC','Gap+ledger+distance AUC'],rows),'',
        '**相同状态反事实：调用位置重合度**','',
        '`Δ/U = |D△P| / |D∪P|`；Jaccard = 1−Δ/U。Knob 列排除固定 guard；全 MISS 列包含两规则共享的 guard。Same-history 列另将两规则放在完全相同的实际历史上，只比较当前一步，去掉独立账本长期相位分化的影响。','']
    rows=[]
    for r in n['pooled_replay']:
        k=r['knob_calls'];rows.append([r['model'],fmt(r['target'],2),names[r['behavior']],k['a'],k['b'],k['union'],
            pct(k['symdiff_over_union']),pct(r['all_calls']['symdiff_over_union']),pct(r['same_history']['symdiff_over_union'])])
    lines += [table(['Model','Dose','Recorded behavior','D K','P K','K union','Knob Δ/U','All MISS Δ/U','Same-history Δ/U'],rows),'',
        '高预算时可跳过的观察更少，因此重合度自然变高。不能把低预算的高 Δ/U 理解成调用发生在完全不同的任务阶段。下表 .32 rep0 同时给出最近调用的 controls 距离（双向最近邻，按调用加权），以及将 D 的归一化距离分数 u 固定为 0.5 后、仍保留反馈账本时的 Δ/U。常数回放是事先定义的消融，不重新拟合 bias。','']
    rows=[]
    const={(r['cell'],r['target'],r['behavior'],r['rep']):r for r in n['constant_score_replay']}
    for r in n['same_state_nearest']:
        if r['target']!=.32 or r['rep']!=0:continue
        c=const[(r['cell'],r['target'],r['behavior'],r['rep'])]
        rows.append([r['cell'],names[r['behavior']],quant(r['nearest_controls']),pct(r['within_10_controls']),pct(c['knob']['symdiff_over_union'])])
    lines += [table(['Cell (.32 rep0)','Behavior','Nearest controls p10/50/90','Within 10 controls','D vs constant-score Δ/U'],rows),'',
        '**配对初始状态上的真实双轨迹：分布比较**','',
        '下表 .32 rep0、knob only。W₁ 是各 episode 两组调用位置经验分布的 Wasserstein 距离，再对匹配 init 平均。精确位置 Δ/U 只在共同 elapsed horizon 内算；W₁ 分布各用完整轨迹。两条轨迹状态与终止长度不同，这一表只说明分布和大致阶段，不能替代上面的同状态比较。所有 dose/repeat 均保存在 JSON。','']
    rows=[]
    for r in n['matched_init_positions']:
        if r['target']!=.32 or r['rep']!=0 or r['kind']!='knob':continue
        d=r['distributions'];rows.append([r['cell'],r['matched_inits'],pct(r['pooled_symdiff_over_union']),fmt(d['controls_wasserstein'].get('mean'),1),
            fmt(d['fraction_wasserstein'].get('mean')),fmt(d['progress_wasserstein'].get('mean')),pct(d['within_10_controls'].get('mean'))])
    lines += [table(['Cell','Matched inits','Exact-position Δ/U','W₁ controls','W₁ episode fraction','W₁ demo progress','Nearest ≤10 controls'],rows),'',
        '**分歧位置的动作误差：.32 rep0 影子臂**','',
        f"标签覆盖四个 size-50 cell，共 {n['counts']['shadow_pilot_looks']:,} looks，其中 {n['counts']['shadow_pilot_eligible']:,} 个非保护观察。size-200 没有这组影子标签，未补造。MISS 用实际策略输出，HIT 用同一 stage-1 上的私有噪声 shadow 输出；都与记录的 corrected cache proposal 比较。主量为前 10 controls、6 个运动通道按冻结 sigma 归一化后的均方误差；前 5 controls 与 gripper 符号分歧另列。它是单样本策略—缓存不一致度，没有动作真值，也不是策略调用的因果收益。",
        '',
        '以下全部在记录状态上重放 D/P，固定同一组 guard。D-only = 距离选、周期跳过；P-only = 距离跳过、周期选。Both 和 neither 保留为参照；完整 gap 分层的样本数、误差与分位数在 JSON 的 shadow_groups 中。','']
    rows=[]
    for cell in sorted({r['cell'] for r in n['shadow_selected_means']}):
        for method in ['distance_feedback','periodic']:
            records={r['selection']:r for r in n['shadow_selected_means'] if r['cell']==cell and r['behavior']==method}
            rows.append([cell,names[method],*[f"{records[k]['n']} / {fmt(records[k]['risk10_mean'])}" for k in ['distance_all','periodic_all','distance_only','periodic_only','both','neither']]])
    lines += [table(['Cell','Behavior','D chosen n/error','P chosen n/error','D-only n/error','P-only n/error','Both n/error','Neither n/error'],rows),'',
        '条件比较按 cell × 行为路径 × task × 实际 gap 精确分层（gap≥12 合并、startup 单列），只在两类都有样本的层内比较，使用 nD·nP/(nD+nP) 权重。区间为 1,000 次物理 init 配对 cluster bootstrap 的描述性 95% 百分位区间，未校正多重比较。每次重采样重算共同支持和权重。不同 cell 的方向不能用合并均值遮盖。','']
    rows=[]
    for r in n['conditional_shadow']['by_cell']:
        rows.append([r['cell'],r['common_support_n'],r['n'],fmt(r['distance_only_standardized_mean']),fmt(r['periodic_only_standardized_mean']),ci(r),fmt(r['ratio'])])
    r=n['conditional_shadow']['pooled']['risk10']
    rows.append(['Pooled',r['common_support_n'],r['n'],fmt(r['distance_only_standardized_mean']),fmt(r['periodic_only_standardized_mean']),ci(r),fmt(r['ratio'])])
    lines += [table(['Cell','Common-support looks','All D/P-only looks','D-only adjusted error','P-only adjusted error','D−P [95% CI]','D/P'],rows),'',
        table(['Sensitivity','Common-support looks','D-only','P-only','D−P [95% CI]'],[[m,r['common_support_n'],fmt(r['distance_only_standardized_mean'],4),fmt(r['periodic_only_standardized_mean'],4),ci(r)] for m,r in n['conditional_shadow']['pooled'].items() if m!='risk10']),'',
        table(['Behavior path (10 controls)','Common-support looks','D-only','P-only','D−P [95% CI]'],[[r['behavior'],r['common_support_n'],fmt(r['distance_only_standardized_mean']),fmt(r['periodic_only_standardized_mean']),ci(r)] for r in n['conditional_shadow']['by_behavior']]),'',
        '**旧版本记录与核验**','']
    rows=[]
    for r in n['arms']:
        if r['source']!='devrec':continue
        z=r['placement'];rp=next(v for v in n['same_state_replay'] if v['arm']==r['arm'] and (v['target']==r['target'] or r['target'] is None and v['target']==.32))
        rows.append([r['cell'],names[r['method']],fmt(r['target'],2),z['episodes'],z['calls'],z['knobs'],quant(z['gap_any_call']),pct(rp['knob_calls']['symdiff_over_union'])])
    lines += [table(['Cell','Rule','Actual dose','Episodes','MISS','Knob','MISS gap p10/50/90','D/P replay Δ/U'],rows),'',
        'off 的反事实栏使用冻结 .32 D/P 配置；两条旧 periodic 路径使用各自实际 dose。其余已规划 devrec 臂无完成 summary/journal，缺失清单保存在 mechanism_inputs.json；没有旧 distance 实测结果可比较。',
        '',
        f"校验：{n['audit']['frozen_files']} 个冻结清单成员前后 SHA256 一致；全部 accepted episodes 与 B pool contract 对齐；MISS=guard+knob；look/MISS 数量对齐原 cost ledger。P 原规则回放、keyed cap、D 原规则回放均为零不一致。cache HIT 动作一致性及 raw-distance 日志对齐误差逐臂保存在 mechanism_inputs.json。pilot 分片 SHA 已由原 gates 审计；本脚本保留 manifest SHA、逐片核验身份与数值，不重复宣称再次散列全部压缩分片。",
        '',
        '复现（仓库根目录，依次提取、分析、生成报告并独立核验；只写本目录新建的 mechanism_* / MECHANISM.md）：',
        '',
        '```bash',
        *[f'taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r13/astra/{name}' for name in ['mechanism_extract.py','mechanism_analyze.py','mechanism_report.py','mechanism_validate.py']],
        '```',
        '',
        '完整数值：`mechanism_numbers.json`；读取来源、arm 完成状态及冻结哈希：`mechanism_inputs.json`；独立对照门禁审计、分母和集合恒等式的核验：`mechanism_validation.json`。`mechanism_looks.pkl` / `mechanism_replayed.pkl` 为仅含 B 数值的派生缓存。本文是机制描述，不改变门禁裁定，也不据此选择参数、dose 或 cell。',
    ]
    return '\n'.join(lines)+'\n'

def main():
    before=frozen();df=pd.read_pickle(HERE/'mechanism_replayed.pkl');n=read(HERE/'mechanism_numbers.json')
    df=exact_lengths(df,n)
    n=supplemental(df,n)
    assert before==frozen()
    n['audit']['report_frozen_unchanged']=True
    report=render(n,df)
    dump(HERE/'mechanism_numbers.json',n)
    df.to_pickle(HERE/'mechanism_replayed.pkl')
    (HERE/'MECHANISM.md').write_text(report)
    print('WROTE MECHANISM.md and supplemented mechanism_numbers.json',flush=True)
    print(json.dumps(clean(n['headline']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
