"""NEW descriptive aggregation of existing MT50 cache journals. CPU only.

Stream task objects from large, pretty-printed EPISODE_PLAN files. No imports
from the repository, no source mutation, no preregistered statistic rerun.
"""
import collections
import hashlib
import json
import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=Path('/data/mw_finetune/eval')
CHAINS={'DP':['csdp_1504'], 'GR00T':['csgr_h5','csgr_h6'],
        'Fast-WAM(partial)':['csfw_k3','csfw_k3w4','csfw_k3w5']}

def stream_tasks(path):
    active=False; buf=[]; start=0
    for line,s in enumerate(path.open(),1):
        if s.startswith('  "tasks": ['):
            active=True; continue
        if not active:continue
        if s.startswith('  ]'):break
        if s.startswith('    {'):buf=[s];start=line
        elif buf:
            buf.append(s)
            if s.startswith('    }'):
                yield start,json.loads(''.join(buf).rstrip().removesuffix(','));buf=[]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def link(p,line=1):return f'[{p.parent.name}/{p.name}:{line}]({p}:{line})'

def main():
    md=['# MT50 缓存长段审核（本次新增描述统计）','',
        '逐 campaign 只纳入 accepted=true、无非空 error 的 done/failed；按原计划逐 uid 对照。',
        'GR00T 续跑只继承 cache_resume_selection.retained_task_uids，不能把所有旧失败尝试混入。',
        '完整 cell 必须有 50任务×50初态、seed7；数字均为百分数。组均值/按任务均值的定义同 MT50_AUDIT.md。',
        '配对键为 (task_name, original_init_idx)，只有两侧完整且同 K 才显示长段对照；未做新推断检验。',
        'Fast-WAM 合并 csfw_k3→w4→w5 的保留 UID；以计划及 study 的实际 num_steps 为准，续跑未齐局。', '',
        '| 模型 | 库大小 | 组 | K | C | n / success | 组均值 / 任务均值 | 完整 | 原始来源 |',
        '|---|---:|---|---:|---:|---|---|---|---|']
    audit={}; comparisons=[]
    for model,chain in CHAINS.items():
        merged={}; sources={}; rejected=collections.Counter()
        for name in chain:
            p=ROOT/name; journal=p/'driver/journal.jsonl'; plan=p/'EPISODE_PLAN.json'
            c=json.loads((p/'campaign.json').read_text())
            if c.get('cache_resume_selection'):
                keep=set(c['cache_resume_selection']['retained_task_uids'])
                merged={uid:r for uid,r in merged.items() if uid in keep}
            eligible={}
            journal_bytes=journal.read_bytes()
            snapshot_time=datetime.datetime.now(datetime.timezone.utc).isoformat()
            for line,s in enumerate(journal_bytes.splitlines(),1):
                try:r=json.loads(s)
                except json.JSONDecodeError:rejected['incomplete_line']+=1;continue
                if r.get('accepted') is not True or r.get('error') or r.get('status')=='error':
                    rejected['ineligible']+=1;continue
                assert r['status'] in ('done','failed') and isinstance(r['success'],bool)
                eligible[r['task_uid']]={'success':r['success'],'journal_line':line,'run':name}
            found=set()
            for lineno,t in stream_tasks(plan):
                uid=t['task_uid']
                if uid not in eligible:continue
                e=t['extra'];assert e['seed']==7
                r=eligible[uid];r.update(size=e['library_size'],group=e['cache_group'],K=e['num_steps'],
                                      C=e['commit_horizon'],task=e['task_name'],idx=t['orig_init_state_idx'],plan_line=lineno)
                assert uid not in merged,(name,uid)
                merged[uid]=r;found.add(uid)
            assert found==set(eligible)
            sources[name]={f:sha(p/f) for f in ['EPISODE_PLAN.json','campaign.json','assets/benchmark.json']}
            sources[name].update(journal_sha256=hashlib.sha256(journal_bytes).hexdigest(),
                                 journal_bytes=len(journal_bytes),snapshot_utc=snapshot_time)
        bench=json.loads((ROOT/chain[-1]/'assets/benchmark.json').read_text())
        difficulty={r[0]:r[2] for r in bench['tasks']}
        cells=collections.defaultdict(dict)
        for uid,r in merged.items():
            key=(r['size'],r['group'],r['K'],r['C']);pair=(r['task'],r['idx'])
            assert pair not in cells[key]
            cells[key][pair]=r
        records=[]
        for key,rows in sorted(cells.items()):
            size,group,k,c=key;n=len(rows);success=sum(r['success'] for r in rows.values())
            counts=collections.Counter(task for task,idx in rows)
            complete=n==2500 and len(counts)==50 and all(v==50 for v in counts.values())
            gm=tm=None
            if complete:
                assert all(sorted(idx for task1,idx in rows if task1==task)==list(range(50)) for task in counts)
                task_sr={task:sum(r['success'] for (t,i),r in rows.items() if t==task)/50 for task in counts}
                by_group=collections.defaultdict(list)
                for task,v in task_sr.items():by_group[difficulty[task]].append(v)
                gm=100*sum(sum(v)/len(v) for v in by_group.values())/len(by_group)
                tm=100*sum(task_sr.values())/50
            runs=sorted({r['run'] for r in rows.values()})
            refs=[]
            for run in runs:
                rr=[r for r in rows.values() if r['run']==run];p=ROOT/run
                refs.extend([link(p/'driver/journal.jsonl',min(r['journal_line'] for r in rr)),
                             link(p/'EPISODE_PLAN.json',min(r['plan_line'] for r in rr))])
            val=f'{gm:.2f} / {tm:.2f}' if complete else '不报未齐 cell 的 SR'
            md.append(f'| {model} | {size} | {group} | {k} | {c} | {n} / {success} | {val} | {complete} | '+ '; '.join(refs)+' |')
            records.append(dict(size=size,group=group,K=k,C=c,n=n,successes=success,group_mean=gm,task_mean=tm,complete=complete,
                                source_lines={run:[r['journal_line'] for r in rows.values() if r['run']==run] for run in runs}))
        for short,long in [('c','ch'),('cr','chr'),('cg','chg'),('crg','chrg')]:
            for a in records:
                if a['group']!=short or not a['complete']:continue
                for b in records:
                    if b['group']!=long or b['size']!=a['size'] or b['K']!=a['K'] or not b['complete']:continue
                    ra=cells[(a['size'],short,a['K'],a['C'])];rb=cells[(b['size'],long,b['K'],b['C'])]
                    assert set(ra)==set(rb)
                    comparisons.append(dict(model=model,size=a['size'],pair=short+'→'+long,K=a['K'],
                        group_a=a['group_mean'],group_b=b['group_mean'],task_a=a['task_mean'],task_b=b['task_mean'],
                        gains=sum(v['success'] and not ra[q]['success'] for q,v in rb.items()),
                        losses=sum(not v['success'] and ra[q]['success'] for q,v in rb.items())))
        audit[model]=dict(cells=records,input_sha256=sources,rejected=dict(rejected))
    md+=['','## 完整的短段→长段对照（同初态配对，描述性）','',
         '| 模型 | 库大小 | 组 | K | 组均值短→长 | 任务均值短→长 | g/l |','|---|---:|---|---:|---|---|---|']
    for r in comparisons:
        md.append(f'| {r["model"]} | {r["size"]} | {r["pair"]} | {r["K"]} | {r["group_a"]:.2f}→{r["group_b"]:.2f} | {r["task_a"]:.2f}→{r["task_b"]:.2f} | {r["gains"]}/{r["losses"]} |')
    (OUT/'MT50_CACHE_AUDIT.md').write_text('\n'.join(md)+'\n')
    (OUT/'mt50_cache_audit.json').write_text(json.dumps(dict(audit=audit,comparisons=comparisons),ensure_ascii=False,indent=2)+'\n')
    print('Wrote cache audit:',{model:len(v['cells']) for model,v in audit.items()},'paired comparisons',len(comparisons))

if __name__=='__main__':main()
