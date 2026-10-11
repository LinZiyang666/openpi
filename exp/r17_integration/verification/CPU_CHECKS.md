# P 本次 CPU verification（2026-10-09）

这是已有接口的实跑检查；没有实现新adapter，没有加载checkpoint，没有使用GPU或真实renderer，也没有操作已运行的训练进程。后续包中的验收清单是拟议测试，不属于本报告已通过项。

| 检查 | 实跑结果 | 证据 / 限定 |
|---|---|---|
| serving identity / per-step / replica helpers / router bundle + MW fakeenv/runner + R17 worker/contract | 89 passed，6 initially skipped，1 deselected；0 failures/errors，9.773s | [cpu_tests.xml:1](/home/weiland/projects/openpi/exp/r17_integration/verification/cpu_tests.xml:1)；6跳过为env_dependent localhost proxy tests，随后补跑全部通过；1排除为真实EGL跨进程初态test。 |
| generic conductor全部 + R16 conductor + GR server topology | 152 passed，0 skipped/failures/errors，12.615s | [conductor_tests.xml:1](/home/weiland/projects/openpi/exp/r17_integration/verification/conductor_tests.xml:1)；synthetic outcomes、fakepolicy/ctl/process mocks，未启动真实模型server/worker。 |
| proxy真实localhostsocket round-trip / backenddown / broadcastfailure / metadatafailure / control→sticky handoff | 6 passed，23 deselected，0 failures/errors，0.153s | [proxy_socket_tests.xml:1](/home/weiland/projects/openpi/exp/r17_integration/verification/proxy_socket_tests.xml:1)；显式 `--run-manual -m env_dependent`，先审查source：仅fakebackends、本地临时port，没有模型或GPU。40s上限。 |

合计 **247 项不同的 CPU 测试通过**；首轮6个skip已在第三组解决。真实EGL test仍未运行；XML累计tests不能简单把首轮skipped当passed重复相加。pytestwarning仅pynvml及Swig弃用提示，不是测试失败。

前两组通用环境：

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  <下面列出的 targets> -p no:cacheprovider \
  --basetemp=<本目录内各自独立的scratch> --junitxml=<本目录内XML> -q
```

第一组targets和参数：

```text
tests/serving/test_policy_identity.py
tests/serving/test_per_step_recorder.py
tests/serving/test_replica_proxy.py
tests/serving/test_router_bundle_rebind.py
tests/exp/metaworld/test_env.py
tests/exp/metaworld/test_runner.py
exp/mw_finetune/tests
-k 'not same_identity_same_initial_state_across_processes'
--basetemp=/home/weiland/projects/openpi/exp/r17_integration/verification/pytest_cpu_20261009
--junitxml=exp/r17_integration/verification/cpu_tests.xml
```

第二组targets和参数：

```text
tests/conductor
exp/offline_search/rounds/r16/test_conductor.py
exp/offline_search/rounds/r16/test_groot_servers.py
--basetemp=/home/weiland/projects/openpi/exp/r17_integration/verification/pytest_conductor_20261009
--junitxml=exp/r17_integration/verification/conductor_tests.xml
```

第三组完整命令：

```bash
timeout -k 5s 40s env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest \
  tests/serving/test_replica_proxy.py -m env_dependent --run-manual \
  -p no:cacheprovider \
  --basetemp=/home/weiland/projects/openpi/exp/r17_integration/verification/pytest_proxy_sockets_20261009 \
  --junitxml=exp/r17_integration/verification/proxy_socket_tests.xml -q
```

边界说明的源码证据：根pytestplugin默认skip manual/env_dependent：[conftest.py:45](/home/weiland/projects/openpi/conftest.py:45)；排除的真实MW test启动EGL subprocess：[test_env.py:243](/home/weiland/projects/openpi/tests/exp/metaworld/test_env.py:243)；proxyenv_dependent只绑定localhostfakebackend：[test_replica_proxy.py:302](/home/weiland/projects/openpi/tests/serving/test_replica_proxy.py:302)；conductor共享fakes：[conftest.py:1](/home/weiland/projects/openpi/tests/conductor/conftest.py:1)；R16测试synthetic：[test_conductor.py:1](/home/weiland/projects/openpi/exp/offline_search/rounds/r16/test_conductor.py:1)；worker测试只验证fake锁和tmp_path snapshot：[test_workers.py:18](/home/weiland/projects/openpi/exp/mw_finetune/tests/test_workers.py:18)。

所有pytest scratch、XML和报告均位于 `exp/r17_integration/verification/`。关闭cacheprovider/pyc防止写代码目录；pytest写入的fixture文件留在本目录，未清理、未删除历史结果。没有访问 `/archive`、`/data/i018`、`/data/idea_sprint`、`/home/weiland/idea-sprint`；localvendor/config读取仅在允许的 `/data/mw_finetune/`，没有读取大型modeltensor。

文档引用核对与版本快照：[SOURCES.json](SOURCES.json) 保存被引用代码/文档的最终审查内容SHA256、行号及对应行文本；[document_validation.json](document_validation.json) 保存所有引用的存在/行号和相对文档链接检查结果。该snapshot是本次交付的证据封存，**不是对训练资产或其他agent之后修改的不变性保证**。交付时若source更新，实施者应按snapshot比对并重新核对相关claim。
