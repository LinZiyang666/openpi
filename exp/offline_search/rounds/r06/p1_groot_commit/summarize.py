"""Render the result JSON files as markdown tables (pasted into HANDBACK.md)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / 'results'


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    out += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def pct(x):
    return f'{100 * x:.2f}%'


def main():
    rc = json.loads((R / 'replay_checks.json').read_text())
    print('## nesting\n')
    print(table(['model', 'cell', 'episodes', 'decisions', 'vision', 'blind', 'tie-rule A == B-off',
                 'deployed A != B-off (decisions / anchors)', 'max abs diff', 'tie audit (A-only -> B-only rows, distance)'],
                [(r['model'], r['cell'], r['episodes'], r['decisions'], r['vision'], r['blind'], 'bit-exact',
                  f"{r['deployed_A_differing_decisions']} / {r['deployed_A_differing_anchors']}",
                  f"{r['max_abs_difference']:.4g}",
                  '; '.join(f"step {t['step']}: {t['A_only']}->{t['B_only']} @ {t['tied_distance']:.6g}"
                            for t in r['tie_audit']) or '-') for r in rc['nesting']]))
    print('\n## parity\n')
    print(table(['cell', 'replay', 'decisions', 'MISS', 'policy tails', 'compared row fields', 'result'],
                [(r['cell'], r['replay'], r['decisions'], r['misses'], r['src'].get('policy_tail', 0),
                  r['compared_row_fields'], 'all equal') for r in rc['parity']]))
    for key, name in (('groot_b', 'GR00T B'), ('pi05_b', 'pi05 C10 (same replay)')):
        print(f'\n## {key} ({name})\n')
        print(table(['cell', 'replay', 'dec', 'vision', 'cache blind', 'MISS', 'tails', 'MISS@end', 'v', 'm',
                     'MISS/vision', 'max blind', 'owner IR/5', 'MISS reasons'],
                    [(r['cell'], r['replay'].split('_')[-1], r['decisions'], r['vision'], r['cache_blind'], r['miss'],
                      r['policy_tail'], r['miss_at_episode_end'], f"{r['vision_share']:.4f}", f"{r['miss_share']:.4f}",
                      f"{r['miss_per_vision']:.4f}", r['max_blind_run'], f"{r['owner_ir_per_five_controls']:.4f}",
                      ', '.join(f'{k} {v}' for k, v in sorted(r['miss_reason'].items())))
                     for r in rc[key]]))
        print()
        print(table(['cell', 'replay', 'vision look reasons', 'all-vision / gap branch', 'terminal rows',
                     'closed (model sign)', 'closed (opposite sign)', 'stock-branch bit changed', 'K4 v / m'],
                    [(r['cell'], r['replay'].split('_')[-1],
                      ', '.join(f'{k}:{v}' for k, v in r['vision_look_reason'].items()),
                      f"{r['all_vision_branch']} / {r['gap_branch']}", r['terminal_rows'],
                      r['terminal_closed_model_sign'], r['terminal_closed_opposite_sign'],
                      r['stock_branch_terminal_bit_changed_by_sign'],
                      f"{r['k4_ledger']['v']:.4f} / {r['k4_ledger']['m']:.4f}") for r in rc[key]]))
    for path in sorted(R.glob('method_tests_*.json')):
        print(f'\n## {path.name}\n')
        d = json.loads(path.read_text())
        if 'mirror' in d:
            print(table(['suite', 'scale', 'episodes', 'decisions', 'mismatches', 'terminal fires P1',
                         'terminal fires stock sign', 'force-MISS verdicts changed'],
                        [(r['suite'], r['scale'], r['episodes'], r['decisions'], r['mismatches'],
                          r['terminal_fires_p1'], r['terminal_fires_stock_sign'],
                          r['force_miss_verdicts_changed_by_fix']) for r in d['mirror']]))
        if 'lifecycle' in d:
            print(table(['suite', 'scale', 'served', 'one-use', 'vetoes'],
                        [(r['suite'], r['scale'], r['served'], r['one_use'], r['vetoes']) for r in d['lifecycle']]))
        if 'fits' in d:
            print(table(['suite', 'scale', 'pi05 fit == C10', 'pi05 m_thr / c_thr', 'GR00T lib / L', 'GR00T m_thr / c_thr'],
                        [(r['suite'], r['scale'], r['pi05_fit_equal_to_deployed_c10'],
                          f"{r['pi05']['m_thr']:.6g} / {r['pi05']['c_thr']:.6g}",
                          f"{r['groot']['lib']} / {r['groot']['L']}",
                          f"{r['groot']['m_thr']:.6g} / {r['groot']['c_thr']:.6g}") for r in d['fits']]))
        for key in ('refusals', 'resign'):
            if key in d:
                print(f'{key}: {len(d[key])} checks')
    p = R / 'plugin_selftests.json'
    if p.exists():
        print('\n## plugin selftests\n')
        print(table(['arm', 'decisions', 'vision', 'blind', 'MISS', 'policy tails', 'eligible', 'stage1', 'dup rejects', 'PASS'],
                    [(r['arm'], r['decisions'], r['vision'], r['blind'], r['miss'], r['policy_tails'],
                      r['eligible_misses'], r['stage1_calls'], r['duplicate_rejections'], r['PASS'])
                     for r in json.loads(p.read_text())]))
    p = R / 'concurrency_summary.json'
    if p.exists():
        print('\n## concurrency\n')
        print(table(['config', 'connections', 'decisions', 'vision', 'blind', 'MISS', 'policy tails', 'peak', 'speedup', 'exact'],
                    [(r['config'], r['connections'], r['decisions'], r['vision'], r['blind'], r['miss'],
                      r['policy_tails'], r['peak'], f"{r['speedup']:.2f}", r['all_rows_actions_verdicts_histories_equal'])
                     for r in json.loads(p.read_text())]))
    p = R / 'arms_validation.json'
    if p.exists():
        print('\n## artifacts\n')
        d = json.loads(p.read_text())
        print(table(['artifact', 'bytes', 'sha256', 'class', 'library', 'rows'],
                    [(a['path'].replace('/home/weiland/trace_runs/os_closed_loop/', ''), a['bytes'], a['sha256'],
                      a['method_class'], a['library'], a['library_rows']) for a in d['artifacts']]))


if __name__ == '__main__':
    main()
