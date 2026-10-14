"""NEW descriptive audit, existing R17 LIBERO journals only; no model imports."""
import collections
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/mw_finetune/eval')
RUNS = ['pl_a2', 'pl_k1a2', 'gl_a3', 'gl_k1a', 'hs5dp_ctl_142812',
        'hs6fw_res_152931', 'formal_cosmos_libero_official2000_20261010_130639_a8ca2043',
        'hs7dpctl_full', 'hs7dpctl_ddim10', 'hs7fwk1_4090c']
CONTROLS = {'pl_k1a2': 'pl_a2', 'gl_k1a': 'gl_a3', 'hs7dpctl_ddim10': 'hs7dpctl_full'}

def cite(p, needle=None):
    line = next((i for i,s in enumerate(p.read_text().splitlines(),1) if needle and needle in s),1)
    return f'[{p.parent.name}/{p.name}:{line}]({p}:{line})'

def main():
    results = {}
    lines = ['# R17 LIBERO journal 核对（本次新增描述统计）', '',
             '只计 accepted=true、无非空 error 的 done/failed 终局；正常任务失败保留。',
             '每个已列 suite 均完整 10 任务×50 官方初态；按 (suite, canonical task, original_init_idx) 对齐。',
             'π/GR 与 DP 后续复跑配对只复核计划键、模型身份、协议身份及 seed；未另做初态字节或渲染像素复核。',
             'g/l 只描述新增成功/失败；不新增显著性或非劣检验。gl_k1a 虽已齐局，但 SUMMARY/RECEIPT 仍为 PARTIAL。', '',
             '| run | K / H_exec | suite | n / success | SR% | g/l（指定对照） | SUMMARY 状态 | 来源 |',
             '|---|---|---|---|---:|---|---|---|']
    for name in RUNS:
        p = ROOT/name
        files = ['EPISODE_PLAN.json', 'assets/benchmark.json', 'SUMMARY.json', 'driver/journal.jsonl']
        plan, bench, summary = [json.loads((p/f).read_text()) for f in files[:3]]
        tasks = {t['task_uid']:t for t in plan['tasks']}
        rows = {}
        excluded = collections.Counter()
        for line,s in enumerate((p/files[-1]).open(),1):
            r=json.loads(s)
            if r.get('accepted') is not True or r.get('error') or r.get('status')=='error':
                excluded['ineligible']+=1; continue
            assert r['status'] in ('done','failed') and isinstance(r['success'],bool)
            t=tasks[r['task_uid']]; e=t['extra']
            key=(t['experiment'], e['task_name'], t['orig_init_state_idx'])
            if key in rows:
                assert rows[key]['success']==r['success']; excluded['duplicate']+=1
            rows[key]={'success':r['success'],'journal_line':line,'K':e['num_steps'],
                       'H':e['commit_horizon'],'model_identity':e['model_identity']}
        results[name]={'rows':rows,'protocol':plan['benchmark_protocol'],'seed':bench['seed'],
                       'status':summary['status'],'excluded':dict(excluded),
                       'sha256':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in files}}
        control=results.get(CONTROLS.get(name))
        if control:
            assert control['protocol']==plan['benchmark_protocol'] and control['seed']==bench['seed']
            assert set(control['rows'])==set(rows)
            assert all(control['rows'][key]['model_identity']==v['model_identity'] for key,v in rows.items())
        for suite in sorted({key[0] for key in rows}):
            sr={key:v for key,v in rows.items() if key[0]==suite}
            assert len(sr)==500
            by_task=collections.defaultdict(list)
            for (_,task,idx) in sr: by_task[task].append(idx)
            assert len(by_task)==10 and all(sorted(v)==list(range(50)) for v in by_task.values())
            k={r['K'] for r in sr.values()}; h={r['H'] for r in sr.values()}
            assert len(k)==len(h)==1
            success=sum(r['success'] for r in sr.values())
            gl='—'
            if control:
                gain=sum(r['success'] and not control['rows'][key]['success'] for key,r in sr.items())
                loss=sum(not r['success'] and control['rows'][key]['success'] for key,r in sr.items())
                gl=f'{gain}/{loss} ({CONTROLS[name]})'
            lines.append(f'| {name} | {next(iter(k))} / {next(iter(h))} | {suite} | 500 / {success} | {success/5:.2f} | {gl} | {summary["status"]} | '+
                         '; '.join([cite(p/'driver/journal.jsonl'),cite(p/'EPISODE_PLAN.json','commit_horizon'),cite(p/'SUMMARY.json','"suites"')])+' |')
    (OUT/'LIBERO_AUDIT.md').write_text('\n'.join(lines)+'\n')
    for v in results.values():v['rows']=[{'key':key,**r} for key,r in sorted(v['rows'].items())]
    (OUT/'libero_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('Audited',len(results),'runs; wrote LIBERO_AUDIT.md and libero_audit.json')

if __name__=='__main__': main()
