"""Collect completed CPU evidence; does not run tests, prepare, or launch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .dryrun import inventory
from .node import atomic
from .regression_report import cases, report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def main():
    manifest = json.loads((HERE/'BEFORE_AFTER_INPUTS.json').read_text())
    missing = [p for paths in manifest.values() for p in paths if not Path(p).is_file()]
    if missing:
        atomic(HERE/'REPORT_PENDING.json',dict(missing=missing))
        print('Report pending: matched dependency replay is queued')
        return
    comparison = report(manifest)
    atomic(HERE/'BEFORE_AFTER.json', comparison)
    preparation = inventory()
    preparation['checked_utc'] = datetime.now(timezone.utc).isoformat()
    atomic(HERE/'PREPARE_DRY_RUN.json', preparation)
    tests = cases(HERE/'TEST_RESULTS_DEFAULT_ON.xml')
    current = json.loads((HERE/'VERIFICATION.json').read_text())
    pins = json.loads((HERE/'PATCH_BASE.json').read_text())
    names = list(pins) + [
        'exp/offline_search/rounds/r17/filter_failed/stage.py',
        'exp/offline_search/rounds/r17/filter_failed/resume.py',
        'exp/offline_search/rounds/r17/filter_failed_knob/formal_a_queue.py',
        'exp/offline_search/rounds/r17/filter_failed_knob/vla_workers.py',
        'exp/offline_search/rounds/r17/filter_failed_knob/vla_shared.py',
        'exp/offline_search/rounds/r16/conductor_runner.py',
        'exp/offline_search/closed_loop/ops/h100/node.py',
        'exp/r17_integration/ops/RUN_CONFIGS.md',
    ]
    names += [str(p.relative_to(ROOT)) for p in sorted(HERE.glob('*.py'))]
    own_pass = all(v['status']=='passed' for v in tests.values()) and bool(tests)
    ops = next(r for r in comparison['suites'] if r['suite']=='ops')
    before_sources = json.loads(Path(ops['before_xml']).with_suffix('.sources.json').read_text())
    after_sources = json.loads(Path(ops['after_xml']).with_suffix('.sources.json').read_text())
    external = {n:before_sources['hashes'][n] == after_sources['hashes'][n]
                for n in before_sources['hashes'] if n not in pins}
    current_matches = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
                       for n,h in after_sources['hashes'].items()}
    target = next((v for k,v in ops['cases'].items()
        if k.endswith('::test_unchanged_single_L_reuses_existing_sealed_artifacts')), None)
    current.update(
        status='PASS' if own_pass and comparison['status']=='PASS' and all(external.values()) and all(current_matches.values()) else 'FAIL',
        updated_utc=datetime.now(timezone.utc).isoformat(),
        new_cpu_tests=dict(passed=sum(v['status']=='passed' for v in tests.values()),
            failed=sum(v['status']=='failed' for v in tests.values()),
            report='TEST_RESULTS_DEFAULT_ON.xml', post_apply_imports=True,
            frozen_r16_import_cwd_checked=True),
        runtime_feedback_checks=dict(
            report='TEST_RESULTS_DEFAULT_ON.xml',
            covers=['periodic recycle stability',
                    'network latency queue proxy rollback',
                    'rate-only descent without server timing',
                    'GPU admission ceiling and later server improvement',
                    'probe windows excluded by prior control state',
                    '150/250 convergence 12/15 minutes',
                    'denied legacy slots sample memory at most every two seconds']),
        before_after_status=comparison['status'], before_after_report='BEFORE_AFTER.json',
        baseline_dependency_snapshot='dependencies_ops_after/PINS.json',
        before_after_table=[{k:r[k] for k in ('suite','before','after','regressions','missing')}
                            for r in comparison['suites']],
        unchanged_ops_external_dependencies=external,
        current_tree_matches_after_test_sources=current_matches,
        reported_hang_case=target,
        sol_campaigns_locally_checked=len(preparation['longseg']),
        sol_campaigns_prepared_or_launched=sum(bool(r['markers']) for r in preparation['longseg']),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope_note='PASS means no before-pass/after-fail regression or after timeout; pre-existing failures remain. '
            'The original 19 files come from verified backups; other current dependencies and tests are shared by both sides.',
        guarded_test='ops/test_direct_transfer.py::test_rsync_daemon_roundtrip_and_owned_cleanup is blocked before rsync spawn in both variants; no daemon was started.',
    )
    followup = HERE/'longseg_check_1018/CPU_PROOF.json'
    if followup.exists() and (HERE/'LONGSEG_UNIT_TESTS.xml').exists():
        sol = json.loads(followup.read_text())
        unit = cases(HERE/'LONGSEG_UNIT_TESTS.xml')
        checks = cases(HERE/'NEW_CHAIN_TESTS.xml')
        queue_rows = [json.loads(p.read_text()) for p in sorted((HERE/'new_chain_dry_run').glob('*.json'))]
        current['replacement_chains'] = dict(
            active_lists=preparation['active_lists'],
            prepare_dry_runs=queue_rows,
            prepare_tests_passed=sum(v['status']=='passed' for v in checks.values()),
            autoscale_tests_passed=current['new_cpu_tests']['passed'],
            sol_unit_tests_passed=sum(v['status']=='passed' for v in unit.values()),
            sol_check_status=sol['status'],sol_checked=len(sol['rows']),
            fresh_local_interpreters=sum(r['status']=='PASS' for r in sol['rows']),
            cached_remote_interpreters=sum(r['status']=='PASS_LOCAL' for r in sol['rows']),
            remote_calls_this_followup=0,formal_preparations=0,launches=0,
            report='longseg_check_1018/CPU_PROOF.json')
        if not (len(queue_rows)==7 and unit and checks and
                all(v['status']=='passed' for v in [*unit.values(),*checks.values()]) and
                all(r['status'] in ('PASS','PASS_LOCAL') for r in sol['rows'])):
            current['status']='FAIL'
        name='exp/offline_search/rounds/r17/longseg_1017/check.py'
        current['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    atomic(HERE/'VERIFICATION.json', current)
    lines=['前后测试对照（通过 / 失败；逐例超时）', '',
           '| 套件 | 备份源码 | 当前源码 | 新增回归 | 当前超时 |',
           '|---|---:|---:|---:|---:|']
    for row in comparison['suites']:
        b,a=row['before'],row['after']
        lines.append(f"| {row['suite']} | {b['passed']} / {b['failed']} | {a['passed']} / {a['failed']} | {len(row['regressions'])} | {a['timeouts']} |")
    lines += ['', f"新增 post-apply CPU 测试：{current['new_cpu_tests']['passed']} 通过；{current['new_cpu_tests']['failed']} 失败。",
              '', '19 份备份均核对 PATCH_BASE.json SHA；只读覆盖导入与源码读取，保留绝对路径和完整当前依赖。两边使用同一组当前测试。ops 每例180秒，其余每例60秒；CUDA禁用，BLAS单线程。',
              '', '初轮期间 cache_study.py 增加了 task_namespace 支持。最终 ops 备份侧已按 dependencies_ops_after/PINS.json 重跑，四个共享依赖与当前侧测试的源码 SHA 一致；初轮备份结果只作排查记录。',
              '', 'H100 真实 rsync daemon 用例两边均 deselect；ops 的 rsync daemon 用例被 CPU guard 在 spawn 前阻断，两边都记失败。这两项不是产品回归结论。',
              '', '完整逐例状态、错误与耗时见 BEFORE_AFTER.json；实际命令、日志、JUnit 路径见 BEFORE_AFTER_INPUTS.json 及各 regression_* 目录。未运行 GPU 验收。']
    if target:
        lines += ['', '原报告卡住用例：备份侧 %.3f 秒，当前侧 %.3f 秒。' %
                  (target['before']['seconds'],target['after']['seconds'])]
    (HERE/'BEFORE_AFTER.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=current['status'], regression_status=comparison['status'],
        tests=current['new_cpu_tests'], longseg=len(preparation['longseg'])),indent=2))


if __name__=='__main__':main()
