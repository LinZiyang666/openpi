# A2：全仓冗余防御普查与拔除

先读 exp/offline_search/rounds/r17/seeds_1023/prompts/COMMON.md（必须遵守：本机禁止任何运算，只改文件和 py_compile，不提交）和 seeds_1023/DESIGN.md「拔的防御」。

## owner 要求（10-23）
「拔掉一切奇怪的没有用的冗余的防御」。项目原则：用最简单的现成办法；哈希核验、源码锁、sha 钉、一次性封条、写死的保守上限，见到就拔；只留真正决定数字对不对、或防止两批数据混在一起的东西。

## 范围
R17 实验运行与建批代码：`exp/r17_integration/ops/`、`exp/r17_integration/search/`、`exp/r17_integration/common/`、`exp/r17_integration/debug/`、`exp/offline_search/rounds/r17/`（scheduler、filter_failed、filter_failed_knob、longseg_1017、longseg_off_1022、knob 等）、`exp/offline_search/rounds/r15/`、`r16/`。不看 tests、不看 frozen 副本目录（`*/followup_*/`、`frozen_runtime/`、`before_tmp/after_tmp/` 这类快照）。

## 必须保留（不准拔）
同卡同步数参照与判格规则；Q3 停滞门（stall_only）及其与 fit 的绑定和度量校验；fit 与 YAML 的 store_digest/commit/binding 一致（`search/cache_recipe.py:160-170`）；RowIndexStore 完整局检查；局数完整性、回执的完成局数/终止错误判定；LAUNCH.json 独占认领（防同一计划重复派发）；拟合防测试集泄漏（`r10/recipe/fitting.py:24-34,37-63`）；旧门（NoProgressGuard/TriggerCommitJudge/guard_only）在服务端的禁用；DP CUDA graph 自检；longseg 补丁锚点；端口冲突检查。

## 归属冲突（这些文件现在被 W1–W4 占用，**只列清单不改**）
W1：`seeds_1023/`（非 vla）、`ops/cache_study.py`、`ops/cache_knob.py`、`filter_failed_knob/{runtime,build,common}.py`、`filter_failed/{audit_counts,stage}.py`、`longseg_1017/{recalibrate,staging,sync_back}.py`、`longseg_off_1022/{build,fit500,workflow}.py`、`ops/formal_a_analysis.py`、`knob/common.py`、evidence.py。
W2：`seeds_1023/vla/`、`filter_failed_knob/{formal_a_queue,vla}.py`、`filter_failed_knob/q3_port_1020/{fit,build,analysis}.py`、`r15/descent/fits.py`、`r16/{analyze,conductor_analysis}.py`。
W3：`debug/library_capture.py`、`search/{teacher_library,cache_recipe}.py`、`ops/{library_campaign,library_staging}.py`、store.py、RUN_CONFIGS.md。
W4：`ops/{formal_eval,formal_finalize,formal_prestage,formal_ports}.py`、`filter_failed_knob/launch.py`。

## 做法
1. 普查范围内所有防御性检查（raise/assert/拒绝/sha/digest/identity 比对/封条/只准某机器/写死上限/超时门等），逐条判断：A 拔（不影响数字、不防混批）；B 留（列在必须保留里或确实防错数据）；C 不确定。
2. 不在占用清单里的文件：A 类直接拔（改成直接放行或删掉检查；被拔检查的结果若别处还读，保持字段兼容）。
3. 占用文件：只写进清单。
4. 交付：`seeds_1023/DEFENSES.md`——已拔（file:line、一句话说明）、待拔（占用文件，file:line、怎么拔）、保留（为什么）、不确定（需要协调者判断的问题）。外加 COMMON.md 要求的交付。
