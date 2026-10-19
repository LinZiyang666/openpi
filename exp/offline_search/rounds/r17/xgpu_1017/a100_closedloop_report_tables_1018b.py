"""Render markdown tables from evidence/a100_closedloop_analysis_1018b.json (CPU)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / 'evidence/a100_closedloop_analysis_1018b.json').read_text())
NAME = {'pi05': 'π0.5 MT50', 'groot': 'GR00T MT50', 'dp': 'DP spatial', 'fastwam': 'Fast-WAM spatial', 'cosmos': 'Cosmos spatial'}


def label(m):
    if m['arm'] == 'pure_libero10':
        return 'DP libero_10 pure'
    return f"{NAME[m['model']]} {m['arm']}"


def ci(r):
    return f"{100*r['difference']:+.2f}；{r['only_a']}/{r['only_b']}；p={r['mcnemar_p']:.3g}；[{100*r['CI95'][0]:+.2f}, {100*r['CI95'][1]:+.2f}]"


print('| 模型 / 臂 | 正式参考卡 | N（块） | 成功 正式/c1/c2 | 同卡 c1−c2（ΔSR pp；b/c；McNemar p；保守95%CI） | A100 c1−正式 | A100 c2−正式 |')
print('|---|---|---:|---|---|---|---|')
for m in d['models']:
    s = m['successes']
    blocks = '+'.join(str(b['episodes']) for b in m['blocks'])
    print(f"| {label(m)} | {m['formal_gpu']} | {m['episodes']} ({blocks}) | {s['formal']}/{s['a100_cold1']}/{s['a100_cold2']} | "
          f"{ci(m['same_card_cold1_vs_cold2'])} | {ci(m['a100_cold1_minus_formal'])} | {ci(m['a100_cold2_minus_formal'])} |")
print()
print('| 模型 / 臂 | N | 同卡不一致 | 跨卡不一致 c1 / c2 | 跨卡−同卡不一致（逐集 exact McNemar p）c1 / c2 |')
print('|---|---:|---:|---:|---|')
for m in d['models']:
    q = m['discordance']; x = m['cross_vs_same_discordance']
    print(f"| {label(m)} | {q['n']} | {q['same_card']} ({100*q['same_card']/q['n']:.1f}%) | {q['cross_cold1']} / {q['cross_cold2']} | "
          f"{x['cold1']['only_first']}:{x['cold1']['only_second']} p={x['cold1']['exact_mcnemar_p']:.3g} / {x['cold2']['only_first']}:{x['cold2']['only_second']} p={x['cold2']['exact_mcnemar_p']:.3g} |")
p = d['pooled']
print(f"\n合计 {p['n']} 对：同卡不一致 {p['same']}，跨卡 c1 {p['cross1']}，c2 {p['cross2']}。")
if d.get('single_cold_blocks'):
    print('\n单冷启动扩展块（未并入逐冷启动序列）：')
    for b in d['single_cold_blocks']:
        print(' ', b['model'], b['arm'], b['episodes'], b['colds'],
              {k: (v['only_a'], v['only_b'], round(100 * v['difference'], 2), [round(100 * x, 2) for x in v['CI95']]) for k, v in b.items() if k.startswith('a100_cold')})
print('\n| 4090 / H100 campaign | suite | N | ΔSR(4090−H100) pp | b/c | exact McNemar p | 保守95%CI pp | 身份差异 |')
print('|---|---|---:|---:|---:|---:|---|---|')
for r in d['existing_4090_vs_h100']:
    miss = f"（4090缺{r['missing_left']}）" if r.get('missing_left') else ''
    print(f"| {r['left_4090']} / {r['right_h100']} | {r['suite']} K{r['num_steps']}{miss} | {r['pairs']} | {r['diff_pp']:+.2f} | {r['only_4090']}/{r['only_h100']} | {r['mcnemar_p']:.3g} | "
          f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] | {', '.join(r['identity_differences']) or 'UID namespace'} |")
print('\n| 不一致率 d（真差0） | 期望情形 N | 80%功效 N |')
print('|---:|---:|---:|')
for r in d['sample_size']:
    print(f"| {100*r['discordance']:.1f}% | {r['n_expected_case']} | {r['n_power80']} |")
