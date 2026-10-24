"""Formal results table and figure: runs/{pure,cache_full,cache,cache_seg}/RESULTS.json (+ HITSPLIT.json from hitsplit.py)
-> runs/REPORT.md, runs/report.png.

  python -m exp.cluster_sim.report --runs /nvme/weiland/openpi/cluster_sim/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {'pure': '全步纯推理（K=10）', 'cache': '完整系统（K=1，缓存）', 'cache_seg': '完整系统 + 长执行段（K=1）', 'cache_full': '缓存系统全步（K=10）'}


def load(runs):
    return {m: {r['robots']: r for r in json.loads((Path(runs) / m / 'RESULTS.json').read_text())}
            for m in ('pure', 'cache_full', 'cache', 'cache_seg') if (Path(runs) / m / 'RESULTS.json').exists()}


def load_split(runs):
    return {m: {r['robots']: {'hit': r['hit_ms'], 'miss': r['miss_ms']} for r in json.loads((Path(runs) / m / 'HITSPLIT.json').read_text())}
            for m in ('cache_full', 'cache', 'cache_seg') if (Path(runs) / m / 'HITSPLIT.json').exists()}


def table(res):
    ns = sorted(set().union(*[set(v) for v in res.values()]))
    cols = {'500 局用时 s（括号内为纯推理用时 / 本组用时）': lambda m, r: f"{r['total_s']:.0f}" + (
                f" ({res['pure'][r['robots']]['total_s'] / r['total_s']:.2f}×)" if 'pure' in res and r['robots'] in res['pure'] else ''),
            '每秒请求数（括号内为全部机器人都在跑的稳态段）': lambda m, r: f"{r['requests_per_s']:.1f} ({r['steady_requests_per_s']:.1f})",
            '服务器端延迟 中位 / p90 / p99 ms': lambda m, r: '{p50:.0f} / {p90:.0f} / {p99:.0f}'.format(**r['latency_ms'])}
    out = []
    for title, cell in cols.items():
        out += [f'### {title}', '', '| N | ' + ' | '.join(LABEL[m] for m in res) + ' |', '|' + '---|' * (1 + len(res))]
        for n in ns:
            out.append(f'| {n} | ' + ' | '.join(cell(m, res[m][n]) if n in res[m] else '' for m in res) + ' |')
        out.append('')
    return '\n'.join(out)


COLORS = {'pure': '#D62728', 'cache_full': '#E69F00', 'cache': '#0072B2', 'cache_seg': '#111111'}  # the three cache series intertwine: blue, orange, black
MARKS = {'pure': 'o', 'cache_full': 's', 'cache': 'o', 'cache_seg': '^'}


def figure(res, path, split=None):
    """split: {series: {N: {'hit': mean ms, 'miss': mean ms}}} adds hit (dotted) and miss (dashed) lines to the latency panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    names = {'pure': 'pure inference (K=10)', 'cache': 'cache system (K=1)', 'cache_seg': 'cache system + long segments (K=1)', 'cache_full': 'cache system, full step (K=10)'}
    for m, rows in res.items():
        ns = sorted(rows)
        kw = dict(color=COLORS[m], lw=2, marker=MARKS[m], ms=4.5)
        axes[0].plot(ns, [rows[n]['requests_per_s'] for n in ns], label=names[m], **kw)
        axes[1].plot(ns, [rows[n]['latency_ms']['mean'] for n in ns], label=names[m], zorder=3, **kw)
        if split and m in split:
            axes[1].plot(ns, [split[m][n]['hit'] for n in ns], color=COLORS[m], lw=1.6, ls=(0, (1, 1.6)))
            axes[1].plot(ns, [split[m][n]['miss'] for n in ns], color=COLORS[m], lw=1.4, ls=(0, (4, 2)))
        axes[2].plot(ns, [rows[n]['total_s'] / 60 for n in ns], label=names[m], **kw)
        if 'pure' in res:
            axes[3].plot(ns, [res['pure'][n]['total_s'] / rows[n]['total_s'] for n in ns], label=names[m], **kw)
    for ax, title, unit in zip(axes, ('server throughput', 'mean request latency (receive -> send)', 'time to finish 500 episodes',
                                      'speed-up vs pure inference (500-episode time)'), ('requests / s', 'ms', 'minutes', 'x')):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('robots N')
        ax.set_ylabel(unit)
        ax.grid(alpha=.25)
        ax.set_ylim(bottom=0)
        ax.spines[['top', 'right']].set_visible(False)
    log_ms(axes[1])
    axes[3].set_ylim(0, 5.5)
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, loc='upper left' if ax in (axes[1], axes[3]) else 'best')
    if split:
        axes[1].add_artist(axes[1].get_legend())
        axes[1].legend(handles=[Line2D([], [], color='#333', lw=2, label='all requests'),
                                Line2D([], [], color='#333', lw=1.6, ls=(0, (1, 1.6)), label='hits'),
                                Line2D([], [], color='#333', lw=1.4, ls=(0, (4, 2)), label='misses')],
                       fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('pi0.5 · LIBERO-10 · one RTX 4090 (single model copy) · 500 recorded episodes per N', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def log_ms(ax):
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.set_yscale('log')
    ax.set_ylim(8, 12000)
    ticks = [10, 30, 100, 300, 1000, 3000]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(t) for t in ticks])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='/nvme/weiland/openpi/cluster_sim/runs')
    args = parser.parse_args()
    res = load(args.runs)
    text = table(res)
    (Path(args.runs) / 'REPORT.md').write_text(text + '\n')
    figure(res, Path(args.runs) / 'report.png', load_split(args.runs))
    print(text)


if __name__ == '__main__':
    main()
