# 正式动态验收：逐例前后回归

原有207个用例：before.xml 207 PASS；after_final.xml 同207个全部PASS，无缺失、无新增失败、无跳过。新增13个真实prepare/plan/ledger/guard/continuation/audit用例全部PASS，总220 PASS。逐例按(classname,name)匹配；原始XML保持不变。首次中间开发失败（peer心跳fixture未更新、未准备目标尚无WORKER_DEPLOYMENT）仅改fixture后纠正，未放松生产检查。

| 改动文件 | 改前 | 改后 | 原207例新增回归 |
|---|---|---|---:|
| trial_lane.py | 人造4/8 trial hard limit及错误owner理由 | 移除；历史helper按physical maxima，正式验收改走新入口 | 0 |
| policy.py | 无可配置node spawn间隔/外部峰值下界 | 默认0保持旧行为，私有技术配置启用 | 0 |
| node.py | 只计ledger，legacy低RSS/PID重启会低估增长 | 外部peak包络+跨PID历史，96GiB reserve，10s共享spawn节拍 | 0 |
| formal_trial.py | 总从父source取runtime SHA/worker payload并attach | 新可选preserve_science使用本campaign已固定runtime/payload；默认分支不变 | 0 |
| legacy_release.py | 不带node内存策略 | 私有stage记录node_config，版本化overlay | 0 |
| legacy.py | AgentGovernor只有默认配置 | 读stage传入的node policy，非私有流程默认不变 | 0 |
| closed_loop/ops/h100/control.py | autoscale driver env无node policy | 仅autoscale分支传node_config，OFF不变 | 0 |
| worker.py | accepted timing无node标签 | 传输元数据补node，不改EpisodeTask或seed | 0 |
| driver.py | 只保存window timing | scheduler accepted后追加小型per-node timing事件 | 0 |
| formal_dynamic.py（新增） | 无正式私有ON入口 | 精确三campaign包装；OFF/dyn全字段proof；R15原生resume | 新增测试PASS |
| formal_dynamic_audit.py（新增） | 无跨正式unit的active/rate验收 | 唯一UID/错误/关闭/kernel/回收验收及60秒per-node CSV | 新增测试PASS |
| 六个*_dyn.sh（新增） | 无 | bash -n全部PASS；原prepare/launch六份SHA未变 | 0 |
| trial_commands.sh / TRIAL.md / HANDOFF_OFF.md | side-lane旧说明 | 撤销旧方案，指向FORMAL_ACCEPTANCE；打印helper只打印不执行 | shell检查PASS |

真实FW formal CPU临时prepare：保留目标c33 runtime、不回退pre-longseg父archive；真实driver打包/不可变本地安装/archived planner，2500个完整task dict相等。remote/transfer/stage transport由CPU-only防护阻止或替身替代；不能称作远端prepare成功。R15源SweepStrategy分片、A-pool rollup、manifest与UID/init均真实构建；20/24档共10000/12000个task dict相等。

新curve测试验证只有scheduler accepted计入每node accepted/min，拒绝的结果不记；端到端audit测试证明UNKNOWN kernel及重复UID不能PASS。旧207项含全局OFF、source payload、不变seed、probe/closure、native legacy回收及公平ledger回归。

GLOBAL开关物理off；未修改/data/eval目录、任何现有wrapper、链脚本或链列表。只读远端：107内存/进程/journal、10.8MiB bounded per_step，以及H100该旧lane进程查询。源证据不移不删。
