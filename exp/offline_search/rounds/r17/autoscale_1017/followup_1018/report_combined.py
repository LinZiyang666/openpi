"""Full comparison of the final combination, including the urgent identity fix."""
import hashlib,json
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import report
from exp.offline_search.rounds.r17.autoscale_1017.regression_run import SUITES
HERE=Path(__file__).resolve().parent
manifest={s:[str(HERE/(s+'_before.xml')),str(HERE/'combined_regression'/(s+'_after.xml'))] for s in SUITES}
value=report(manifest)
value['baseline']='Previously accepted post-apply tree: 9 immutable source snapshots plus unchanged original core WorkerLoop. Final after includes physical maxima, fairness and generation identity fix.'
core='src/openpi/conductor/worker.py'
before=hashlib.sha256((HERE/'generation_before/worker.py').read_bytes()).hexdigest()
after=hashlib.sha256((HERE/'generation_candidate/worker.py').read_bytes()).hexdigest()
for row in value['suites']:
    pins=[json.loads(Path(p).with_suffix('.sources.json').read_text())['hashes'] for p in manifest[row['suite']]]
    row['generation_source_verified']=pins[0].pop(core)==before and pins[1].pop(core)==after
    row['unchanged_other_shared_dependencies']=pins[0]==pins[1]
    if not row['generation_source_verified'] or not row['unchanged_other_shared_dependencies']:value['status']='FAIL'
(HERE/'BEFORE_AFTER_INPUTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'BEFORE_AFTER.json').write_text(json.dumps(value,indent=2)+'\n')
lines=['物理上限、公平共享与代际身份修复：最终前后对照','',
       '| 套件 | 修改前：通过 / 失败 | 修改后：通过 / 失败 | 新回归 | 超时 |',
       '|---|---:|---:|---:|---:|']
for r in value['suites']:
    a,b=r['before'],r['after']
    lines.append(f"| {r['suite']} | {a['passed']} / {a['failed']} | {b['passed']} / {b['failed']} | {len(r['regressions'])} | {b['timeouts']} |")
lines+=['','基线为此前已接受的共享树；最终候选同时包含9个autoscale模块、fairness.py和核心WorkerLoop代际身份修复。逐例比较与源码SHA均记录；其余共享依赖须相同。',
    '', '每例限时：ops180秒，其余60秒。CPU guard禁止真实远端、GPU、传输命令；H100真实rsync daemon用例两侧排除，ops同类用例两侧被guard阻止并记作既有失败。',
    '', '先前ops_after_final进程因用户中断而终止在25%，不作为完整结果。本表使用combined_regression/下完整重跑，不拼接截断结果。历史19文件补丁对照仍保留于上级BEFORE_AFTER.md。']
(HERE/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n')
print(value['status'])
for r in value['suites']:print(r['suite'],r['before']['passed'],r['before']['failed'],'->',r['after']['passed'],r['after']['failed'],'regressions',len(r['regressions']),'timeouts',r['after']['timeouts'])
