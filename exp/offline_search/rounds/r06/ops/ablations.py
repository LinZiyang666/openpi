#!/usr/bin/env python3
"""Recompute the R6 ablation tables (ABLATIONS.md §1–§3) from each arm's summary.json and client journal.

Journals are taken after the exception-episode repair. A / B references are the three replicates of PAPER_AB.md
(ops/paper_ab.py:arms). Each ablation arm is paired (exact McNemar on (task, init)) against every replicate.
"""
import importlib.util
import os

spec = importlib.util.spec_from_file_location('paper_ab_mod', os.path.join(os.path.dirname(__file__), 'paper_ab.py'))
pab = importlib.util.module_from_spec(spec)
import sys
_argv, sys.argv = sys.argv, [sys.argv[0]]
_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
spec.loader.exec_module(pab)
sys.stdout = _stdout
sys.argv = _argv

MODELS = {'pi05': 'p', 'groot': 'g'}
LABEL = {'l10_50': 'LIBERO-10, 50', 'l10_500': 'LIBERO-10, 500', 'sp_50': 'Spatial, 50', 'sp_500': 'Spatial, 500'}


def sr(o):
    return sum(o.values()) / len(o)


def ir(model, c):
    return pab.C1[model] * c['v'] + (1 - pab.C1[model]) * c['m']


def pairs(o, refs):
    out = []
    for r, _ in refs:
        k = set(o) & set(r)
        plus = sum(1 for x in k if o[x] and not r[x])
        minus = sum(1 for x in k if r[x] and not o[x])
        out.append((plus, minus, pab.mcnemar(plus, minus)))
    return out


def pfmt(ps):
    return ' / '.join(f'{p:.2g}' for _, _, p in ps)


def table(kind, cells):
    rows = []
    for model, cell in cells:
        A, B = pab.arms(model, cell)
        la, lb = [pab.load(x) for x in A], [pab.load(x) for x in B]
        am = sum(sr(o) for o, _ in la) / 3
        arm = f'r06_abl:r6p2_{kind}_{MODELS[model]}_{cell}'
        o, c = pab.load(arm)
        s = sr(o)
        rows.append((model, cell, am, s, ir(model, c), pairs(o, la), len(o)))
    return rows


def main():
    cells = [(m, c) for m in ('pi05', 'groot') for c in ('l10_50', 'l10_500', 'sp_50', 'sp_500')]
    for kind, title in (('identity', '§1 identity metric'), ('direct', '§2 direct token PCA')):
        print(f'### {title}')
        print('| model | cell | A (3-run mean) | ablation | Δ (pp) | owner IR | p vs A replicates (+/− vs rep 1) | n |')
        print('|---|---|---|---|---|---|---|---|')
        for model, cell, am, s, i, ps, n in table(kind, cells):
            print(f"| {'π0.5' if model == 'pi05' else 'GR00T'} | {LABEL[cell]} | {am:.3f} | {s:.3f} | {100 * (s - am):+.1f} | "
                  f"{i:.3f} | {pfmt(ps)} (+{ps[0][0]}/−{ps[0][1]}) | {n} |")
        print()
    print('### §3 trigger leave-one-out (50-episode libraries)')
    print('| model | cell | guard removed | SR | owner IR | p vs B replicates | p vs A replicates | n |')
    print('|---|---|---|---|---|---|---|---|')
    for model in ('pi05', 'groot'):
        for cell in ('l10_50', 'sp_50'):
            A, B = pab.arms(model, cell)
            la, lb = [pab.load(x) for x in A], [pab.load(x) for x in B]
            bm = sum(sr(o) for o, _ in lb) / 3
            bi = sum(ir(model, c) for _, c in lb) / 3
            print(f"| {'π0.5' if model == 'pi05' else 'GR00T'} | {LABEL[cell]} | — (B, 3-run mean) | {bm:.3f} | {bi:.3f} | — | — | 1500 |")
            for g in ('stuck', 'terminal', 'overtime', 'no_progress'):
                o, c = pab.load(f'r06_abl:r6p2_{g}_{MODELS[model]}_{cell}')
                print(f"| {'π0.5' if model == 'pi05' else 'GR00T'} | {LABEL[cell]} | {g.replace('_', '-')} | {sr(o):.3f} | "
                      f"{ir(model, c):.3f} | {pfmt(pairs(o, lb))} | {pfmt(pairs(o, la))} | {len(o)} |")


if __name__ == '__main__':
    main()
