"""Render cross-trajectory-only Spatial and LIBERO-10 pruning diagnostics.

Consumes suite outputs from prune_feasibility.py, writes a joint Chinese report
and standalone figures. Never reads or modifies a production cache pickle.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


SUITES = {'libero_spatial': 'Spatial', 'libero_10': 'LIBERO-10'}
ROOT = Path('exp/rit_pareto/analysis/trajectory_prune_20260911')


def _read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _index(summary, field):
    return {row['label']: row for row in summary[field]}


def _table(summary, thresholds):
    prune, loo, timing = (_index(summary, key) for key in ['prune', 'loo_results', 'latency'])
    lines = ['| 融合分数阈值 | 保留点 | 删点比例 | 剩余长度减少¹ | 动作 RMSE 变化² | 检索中位 ms³ |',
             '|---|---:|---:|---:|---:|---:|',
             f'| 原库 | {summary["n_entries"]:,} | 0% | 0 | 0% | {timing["baseline"]["median"]:.3f} |']
    for threshold in thresholds:
        label = f'cross_{threshold:g}'
        p, l = prune[label], loo[label]
        ms = f'{timing[label]["median"]:.3f}' if label in timing else '未测'
        lines.append(f'| {threshold:g} | {p["kept"]:,} | {p["removed_pct"]:.1f}% | '
                     f'{l["mean_remaining_reduction"]:.3f} | {l["action_rmse_change_pct"]:+.2f}% | {ms} |')
    return '\n'.join(lines)


def main():
    """Generate the joint report and verify every recorded pair is cross-trajectory."""
    summaries = {suite: json.loads((ROOT / suite / 'summary.json').read_text()) for suite in SUITES}
    validations = {}
    comparison = []
    for suite, summary in summaries.items():
        assert summary['suite'] == suite
        assert all(row['cross_only'] and row['deleted_same_trajectory_pct'] == 0 for row in summary['prune'])
        assert all(row['same_trajectory'] == 'False' for row in _read_csv(ROOT / suite / 'replacement_pairs.csv'))
        validations[suite] = json.loads((ROOT / suite / 'direct_strategy_validation.json').read_text())
        p, loo, timing = (_index(summary, key) for key in ['prune', 'loo_results', 'latency'])
        baseline_remaining = summary['loo_results'][0]['mean_selected_remaining']
        for label, row in p.items():
            measured = timing.get(label)
            comparison.append(dict(suite=suite, threshold=row['threshold'], removed_pct=row['removed_pct'],
                                   kept=row['kept'], remaining_reduction=loo[label]['mean_remaining_reduction'],
                                   remaining_reduction_pct=loo[label]['mean_remaining_reduction'] / baseline_remaining * 100,
                                   action_rmse_change_pct=loo[label]['action_rmse_change_pct'],
                                   median_search_ms=measured['median'] if measured else None,
                                   search_reduction_pct=100 * (1 - measured['median'] / timing['baseline']['median']) if measured else None))
    (ROOT / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    spatial, l10 = summaries['libero_spatial'], summaries['libero_10']
    sl, ll = _index(spatial, 'loo_results'), _index(l10, 'loo_results')
    source = '''# Spatial / LIBERO-10：跨轨迹短剩余优先剪枝

日期：2026-09-11。**比较对象必须来自同一 task、不同 trajectory**。报告、图表、脚本与结果表仅保留这一规则；原库未改。

正式流程固定为 **prune → 重跑 warmup 标定 → eval**。本次是原库上的先期分析，不以旧 RIT 阈值直接套新库的结果判断可行性。

## 结论

两套库都有冗余可剪。LIBERO-10 在相近的动作偏差增幅下，剪点比例及向短剩余轨迹偏移的幅度更大，值得优先进入重标定后的闭环验证。当前证据涉及候选压缩、CPU 检索与动作保真代理，尚未证明 SR 或真实任务完成时间改善。

## 正式检索口径

直接调用 `WeightedScoreSumKnnStrategy → CacheStorage → InMemoryBackend`。两路视觉 cosine、robot_state L2，沿用各 suite 原 zscore–tanh 标定及融合权重；prompt/vision_2 禁用，task_scoped=True、depth=1、step_filter=all。只在枚举候选时扩大 top_k，正式计时用 top_k=1。

| Suite | 成功轨迹数 / 点数 | vision_0 / vision_1 / state 权重 | 自查询分数 | 跨轨迹最近邻分数中位 |
|---|---:|---|---:|---:|
'''
    for suite, summary in summaries.items():
        weights = summary['config_contents']['keys']
        weight_text = ' / '.join(str(weights[key]['weight']) for key in ['vision_0', 'vision_1', 'robot_state'])
        source += (f'| {SUITES[suite]} | {summary["n_trajectories"]} / {summary["n_entries"]:,} | {weight_text} | '
                   f'{summary["self_score"]["median"]:.6f} | {summary["nearest_cross_trajectory_score"]["median"]:.6f} |\n')
    source += '''
**阈值是各自正式融合分数，不是 cosine，也不是跨 suite 通用距离。** L10 的权重与标定使分数更靠近 1，不能直接套 Spatial 的 0.98。

## Spatial

'''
    source += _table(spatial, [.987, .985, .9825, .98, .975])
    source += '\n\n## LIBERO-10\n\n'
    source += _table(l10, [.9985, .998, .9975, .997, .996, .995, .99, .98])
    source += '''

¹ 整条轨迹留出：先排除查询所在整条轨迹，再对剩余库剪枝。剩余长度是选中点在原始轨迹上的后继点数，不是剪后点数，也不是实测 rollout 时长。每个决策一般执行 5 个控制步，末个决策可能提前结束，故不直接换算完成时长。

² 相对留出轨迹原 teacher 动作的 RMSE；取 chunk 前 5 步、前 7 个有效维度，归一化空间。**偏差增加不等于成功率下降**，不同动作也可能更高效。查询来自既有成功轨迹，而不是新的独立闭环 rollout。

³ 同机 CPU、4 线程、100 个查询 × 7 轮、warm cache、每轮随机方案顺序；中位耗时仅含正式 SearchStrategy，不含 key 构造、GPU、网络与仿真。细小计时差异不应过度解读；删点比例不等于 pkl 字节压缩率或端到端提速。

## 代表点的不确定性与控制

'''
    for suite, label in [('libero_spatial', 'cross_0.98'), ('libero_10', 'cross_0.998'), ('libero_10', 'cross_0.997')]:
        summary = summaries[suite]
        row = _index(summary, 'loo_results')[label]
        base = summary['loo_results'][0]['mean_selected_remaining']
        a, b = row['remaining_reduction_ci95']
        c, d = row['action_rmse_change_pct_ci95']
        source += (f'- **{SUITES[suite]} / {row["threshold"]:g}**：原平均剩余 {base:.3f} 点；'
                   f'减少 {row["mean_remaining_reduction"]:.3f} 点（95% 区间 {a:.3f}–{b:.3f}），'
                   f'约 {row["mean_remaining_reduction"] / base * 100:.1f}%；'
                   f'RMSE 变化 {row["action_rmse_change_pct"]:+.2f}%（{c:+.2f}%–{d:+.2f}%）。'
                   f'各留出折同数量随机删除、20 个种子的平均 RMSE 变化为 {row["random_rmse_change_pct_mean"]:+.2f}%。\n')
    source += '''
区间按整条查询轨迹 bootstrap 4,000 次，是探索性描述；未校正多阈值选择，不是正式显著性结论。

## 剪枝定义

1. 只在同任务、不同轨迹的点之间比较，取正式融合分数 ≥ 阈值为近邻。
2. 剩余长度 = 原完整轨迹中该点之后的点数，终点为 0。
3. 从短剩余到长剩余处理；只有已保留的、直接相近且严格更短的跨轨迹点能使当前点删除。长度相等不删，不把已删点当覆盖证据。
4. 单点独立删除，不递归删除其后续轨迹；剪枝过程不重算原始剩余长度。所有原终点保留。
5. 留出诊断每折先排除整个查询 trajectory，再执行上述剪枝，避免查询自己决定保留集。

## 标定与闭环验证

剪后每份库重新 warmup 标定并 eval，使用同一评测初始状态池。比较 SR、各档比例、端到端时延、成功 episode 控制步数；完成长度同时报告共同成功初始状态的配对差，防止失败掉的长轨迹造成幸存者偏差。应按相同实测计算预算或相同 SR 损失比较，不能以旧阈值直接套新库。

当前相似度使用剪枝前正式标定，这定义了“哪些点足够近”。重标定后的命中率和全程速度本次未测。原 RIT per_step 日志的 winner 覆盖仅记录哪些既有决策会受影响，不当作反事实 SR。

`.pkl` 中的 `prev_ids/next_ids` 和快照要保持可解释：可先屏蔽检索候选，同时保留链与原长度；不能递归删除后续、也不能把删点后的链直接拼成新示范。当前只生成保留 ID 清单，没有生成上线 pkl。

## 数据身份与校验

'''
    for suite, summary in summaries.items():
        h5_rows = _read_csv(ROOT / suite / 'source_h5_audit.csv')
        omitted = [r for r in h5_rows if r['in_library'] == 'False']
        present = [r for r in h5_rows if r['in_library'] == 'True']
        success_count = sum(r['success'] == 'True' for r in present)
        v = validations[suite]
        source += (f'### {SUITES[suite]}\n\n'
                   f'- 本地库 `{summary["library"]}`，SHA256 `{summary["library_sha256"]}`。\n'
                   f'- 正式配置 `{summary["config"]}`。\n'
                   f'- RIT export 指向服务器同名库，SHA256 `{summary["server_library_sha256"]}`；'
                   '本地/服务器内容一致性来自数据台账，本轮未连接服务器重新取证。\n'
                   f'- 原轨迹长度：最短 {summary["trajectory_lengths"]["min"]:.0f}、'
                   f'中位 {summary["trajectory_lengths"]["median"]:.0f}、最长 {summary["trajectory_lengths"]["max"]:.0f} 点。\n'
                   f'- 本地源 HDF5 {len(h5_rows)} 个；库内对应 {len(present)} 个，success=True 为 {success_count} 个；'
                   f'未入库 {len(omitted)} 个。\n')
        for row in omitted:
            source += f'  - `{row["file"]}`：success={row["success"]}，{row["num_steps"]} 个决策点。\n'
        source += (f'- 所有删除配对跨轨迹、同任务、原长度严格缩短、直接代表点仍保留；原链完整性通过。\n'
                   f'- {v["direct_pruned_strategy_queries"]} 次剪后真实 SearchStrategy 调用，winner 一致率 '
                   f'{v["winner_agreement"] * 100:.0f}%，最大分数误差 {v["max_score_error"]:.2g}。\n'
                   f'- 原 pkl 前后 SHA 一致。结果明细见 [{suite}/summary.json]({suite}/summary.json)。\n\n')
    source += '''## 复现与产物

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m exp.rit_pareto.analysis.prune_feasibility --suite libero_spatial
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -m exp.rit_pareto.analysis.prune_feasibility --suite libero_10
MPLCONFIGDIR=/tmp/openpi-prune-mpl .venv/bin/python -m exp.rit_pareto.analysis.render_prune_feasibility
```

每个 suite 子目录含配置/哈希、配对分数、跨轨迹删除证据、保留 ID、逐查询留出预测、每任务保留量、原正式 winner 暴露、逐次计时与策略一致性校验。根目录 `comparison.json` 汇总两套库，旧的同轨迹比较结果已删除。

![Cross-trajectory pruning diagnostics](prune_diagnostics.png)
'''
    (ROOT / 'report.md').write_text(source)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.1), constrained_layout=True)
    for suite, color in [('libero_spatial', '#287d8e'), ('libero_10', '#ba433c')]:
        summary = summaries[suite]
        ps = [r for r in summary['prune'] if r['removed_pct'] <= 73]
        ls = _index(summary, 'loo_results')
        x = [r['removed_pct'] for r in ps]
        axes[0].plot(x, [ls[r['label']]['action_rmse_change_pct'] for r in ps], 'o-', color=color, label=SUITES[suite])
        axes[1].plot(x, [ls[r['label']]['mean_remaining_reduction'] for r in ps], 'o-', color=color, label=SUITES[suite])
        timing = _index(summary, 'latency')
        ps = [r for r in ps if r['label'] in timing]
        axes[2].plot([0] + [r['removed_pct'] for r in ps], [0] + [100 * (1 - timing[r['label']]['median'] / timing['baseline']['median']) for r in ps],
                     'o-', color=color, label=SUITES[suite])
    axes[0].set(ylabel='Action RMSE change (%)', title='(a) Action fidelity proxy')
    axes[1].set(ylabel='Mean remaining-point reduction', title='(b) Short-continuation bias')
    axes[2].set(ylabel='Median search latency reduction (%)', title='(c) Local CPU search')
    for ax in axes:
        ax.set_xlabel('Removed library entries (%)')
        ax.axhline(0, color='#cccccc', linewidth=.7, zorder=0)
    axes[0].legend(frameon=False)
    fig.suptitle('Cross-trajectory pruning only | Pi0.5 | Spatial: 49 trajectories / 1,018 points; LIBERO-10: 50 / 2,640\n'
                 'Offline diagnostics; formal evaluation requires fresh warmup calibration', fontsize=12)
    fig.savefig(ROOT / 'prune_diagnostics.png', dpi=180)
    fig.savefig(ROOT / 'prune_diagnostics.pdf')
    print(ROOT / 'report.md')


if __name__ == '__main__':
    main()
