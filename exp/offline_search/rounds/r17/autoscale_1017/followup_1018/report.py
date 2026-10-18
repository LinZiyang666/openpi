"""Compare this follow-up against the previously accepted, applied tree."""
import hashlib,json
from pathlib import Path
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import report
from exp.offline_search.rounds.r17.autoscale_1017.regression_run import SUITES
HERE=Path(__file__).resolve().parent
manifest={s:[str(HERE/(s+'_before.xml')),str(HERE/(s+('_after_final.xml' if s=='ops' else '_after.xml')))] for s in SUITES}
(HERE/'BEFORE_AFTER_INPUTS.json').write_text(json.dumps(manifest,indent=2)+'\n')
value=report(manifest)
value['baseline']='Previously accepted post-apply tree; 9 immutable SHA-verified runtime snapshots via read-only source overlay. Existing tests and all other shared dependencies are identical in both variants.'
for row in value['suites']:
    pins=[json.loads(Path(p).with_suffix('.sources.json').read_text())['hashes'] for p in manifest[row['suite']]]
    row['unchanged_shared_dependency_hashes']=pins[0]==pins[1]
    if pins[0]!=pins[1]:value['status']='FAIL'
(HERE/'BEFORE_AFTER.json').write_text(json.dumps(value,indent=2)+'\n')
lines=['本次 follow-up 前后对照：物理默认上限与双 lane 公平共享','',
       '“前”是 coordinator 已接受的默认开启树；“后”是本次冻结候选。原19文件补丁的历史对照保留在上级 BEFORE_AFTER.md。','',
       '| 套件 | 修改前：通过 / 失败 | 修改后：通过 / 失败 | 新回归 | 超时 |',
       '|---|---:|---:|---:|---:|']
for r in value['suites']:
    a,b=r['before'],r['after']
    lines.append(f"| {r['suite']} | {a['passed']} / {a['failed']} | {b['passed']} / {b['failed']} | {len(r['regressions'])} | {b['timeouts']} |")
lines+=['','每例限时：ops 180秒，其余60秒。CPU guard 禁止实际远端/传输/GPU命令。H100真实rsync daemon用例两侧均排除；ops中的同类用例被guard阻断，记作两侧失败。失败数含此前已存在的失败，不代表全仓测试通过。',
        '', '第一轮候选ops开始后补上了跨lane内存读取失败的暂停准入；其结果仅作诊断。最终ops_after_final.xml是在冻结后的最终源码上完整重跑，其余七组在该修复之后启动。',
        '', '逐例状态、耗时、失败原因及源码一致性见 BEFORE_AFTER.json；基线源码和SHA见 BEFORE_SOURCES.json，候选命令和SHA见 REGRESSION_COMMANDS.json。']
(HERE/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n')
print(value['status'])
for r in value['suites']:print(r['suite'],r['before']['passed'],r['before']['failed'],'->',r['after']['passed'],r['after']['failed'],'regressions',len(r['regressions']),'timeouts',r['after']['timeouts'])
