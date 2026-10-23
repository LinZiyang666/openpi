"""Formal results table and figure: runs/{pure,cache}/RESULTS.json -> runs/REPORT.md, runs/report.png.

  python -m exp.cluster_sim.report --runs /nvme/weiland/openpi/cluster_sim/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {'pure': '全步纯推理（K=10）', 'cache': '完整系统（K=1，缓存）'}


def load(runs):
    return {m: {r['robots']: r for r in json.loads((Path(runs) / m / 'RESULTS.json').read_text())}
            for m in ('pure', 'cache') if (Path(runs) / m / 'RESULTS.json').exists()}


def table(res):
    ns = sorted(set().union(*[set(v) for v in res.values()]))
    lines = ['| N | ' + ' | '.join(f'{LABEL[m]} {c}' for m in res for c in ('请求/秒', '延迟中位/p90/p99 ms', '500 局用时 s')) + ' |',
             '|' + '---|' * (1 + 3 * len(res))]
    for n in ns:
        cells = []
        for m in res:
            r = res[m].get(n)
            if r is None:
                cells += ['', '', '']
                continue
            lat = r['latency_ms']
            cells += [f"{r['requests_per_s']:.1f}", f"{lat['p50']:.0f} / {lat['p90']:.0f} / {lat['p99']:.0f}", f"{r['total_s']:.0f}"]
        lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def figure(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {'pure': '#c0504d', 'cache': '#2f6db5'}
    names = {'pure': 'pure inference (K=10)', 'cache': 'cache system (K=1)'}
    for m, rows in res.items():
        ns = sorted(rows)
        axes[0].plot(ns, [rows[n]['requests_per_s'] for n in ns], 'o-', color=colors[m], label=names[m])
        axes[1].plot(ns, [rows[n]['latency_ms']['p50'] for n in ns], 'o-', color=colors[m], label=f'{names[m]} p50')
        axes[1].plot(ns, [rows[n]['latency_ms']['p99'] for n in ns], 'o--', color=colors[m], alpha=.6, label=f'{names[m]} p99')
        axes[2].plot(ns, [rows[n]['total_s'] / 60 for n in ns], 'o-', color=colors[m], label=names[m])
    for ax, title, unit in zip(axes, ('server throughput', 'request latency (receive -> send)', 'time to finish 500 episodes'),
                               ('requests / s', 'ms', 'minutes')):
        ax.set_title(title)
        ax.set_xlabel('robots N')
        ax.set_ylabel(unit)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle('pi0.5 · LIBERO-10 · one RTX 4090 (single model copy)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='/nvme/weiland/openpi/cluster_sim/runs')
    args = parser.parse_args()
    res = load(args.runs)
    text = table(res)
    (Path(args.runs) / 'REPORT.md').write_text(text + '\n')
    figure(res, Path(args.runs) / 'report.png')
    print(text)


if __name__ == '__main__':
    main()
