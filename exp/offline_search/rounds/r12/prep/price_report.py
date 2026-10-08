"""Aggregate retained raw timings; owner remainder is fixed, never repriced."""
import json
from pathlib import Path

from bench_latency import summarize

HERE = Path(__file__).resolve().parent


def main():
    pi = json.loads((HERE / 'pi05_latency.json').read_text())
    gl = json.loads((HERE / 'groot_l10_latency.json').read_text())
    gs = json.loads((HERE / 'groot_spatial_latency.json').read_text())
    cases = [('pi05', pi['rows'], .152, 'stock_full'), ('groot_l10', gl['rows'], .148, 'full'),
             ('groot_spatial', gs['rows'], .148, 'full'), ('groot_both', gl['rows']+gs['rows'], .148, 'full')]
    report = {}
    for name, rows, basis, denominator in cases:
        timings = {k: summarize([v for r in rows for v in r['timings'][k]['samples_ms']]) for k in rows[0]['timings']}
        normal = timings[denominator]['median_ms']
        report[name] = dict(owner_basis=basis, normalization_full_ms=normal,
            prices={k: (1-basis if k == 'policy_remainder' else basis * t['median_ms']/normal) for k,t in timings.items()},
            timings=timings, observations=len(rows), samples_per_path=sum(len(r['timings']['full']['samples_ms']) for r in rows))
    (HERE / 'price_table.json').write_text(json.dumps(report, indent=1) + '\n')
    lines = ['| Path | π0.5 ms (IQR), price | GR00T L10 ms (IQR), price | GR00T Spatial ms (IQR), price | GR00T pooled ms (IQR), price |',
             '| --- | ---: | ---: | ---: | ---: |']
    for path in ('stock_full','full','wrist_only','third_only','completion_after_wrist','completion_after_third','policy_remainder'):
        values=[]
        for name,_,_,_ in cases:
            case=report[name]
            if path not in case['timings']: values.append('—');continue
            t=case['timings'][path]
            values.append(f'{t["median_ms"]:.3f} ({t["iqr_ms"]:.3f}), {case["prices"][path]:.6f}')
        lines.append('| '+path+' | '+' | '.join(values)+' |')
    (HERE/'PRICE_TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
