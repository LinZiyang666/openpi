"""Check report artifacts and exercise the future A-log reader on allowed supplements.

Only the explicit two supplementary A arms are opened for reader validation;
their outcomes never enter B or the lean A estimate. No synthetic rollouts.
"""
from __future__ import annotations

import csv as csvlib

from final_common import *
from final_test_a import measure, SUPPLEMENT


def main():
    frozen = verify_frozen()
    b = read(HERE / 'final_b_numbers.json')
    a = read(HERE / 'final_test_a_numbers.json')
    cdf = pd.read_csv(HERE / 'final_b_gap_cdf.csv')
    summaries = pd.read_csv(HERE / 'final_b_gap_summary.csv')
    groups = ['cell', 'target', 'rep', 'method', 'kind', 'unit']
    checked = 0
    for key, z in cdf.groupby(groups):
        assert z.gap.iloc[0] == 0 and np.all(z.gap.diff().dropna() == 1)
        assert z.total.nunique() == 1 and z['count'].sum() == z.total.iloc[0]
        assert np.allclose(z['count'].cumsum()/z.total, z.cdf)
        assert z.cdf.iloc[-1] == 1
        if key[-1] == 'real_looks':
            s = summaries
            for col, val in zip(groups[:-1], key[:-1]):
                s = s[s[col] == val]
            assert len(s) == 1
            assert s.n.iloc[0] == z.total.iloc[0]
            assert np.isclose(s['mean'].iloc[0], (z.gap*z['count']).sum()/z.total.iloc[0])
        checked += 1
    # Repeating the same identities must not create artificial bootstrap precision.
    original = [dict(suite='validation', task=0, init=i, success_d=int(i % 3 == 0), success_p=int(i % 4 == 0))
                for i in range(20)]
    one, repeated = paired_stats(original), paired_stats(original * 4)
    assert one['ci95'] == repeated['ci95'] and one['difference'] == repeated['difference']
    with record(HERE / 'test_grid.csv').open() as f:
        specs = {r['arm']: r for r in csvlib.DictReader(f)}
    reader_checks = []
    for name in SUPPLEMENT[1:3]:
        r, eps, curve = measure(specs[name], supplementary=False)
        assert r['status'] == 'complete', r.get('reason')
        assert r['gap']['n'] == r['M'] - r['startup_calls']
        assert len(eps) == r['episodes']
        for unit in ('real_looks', 'controls'):
            z = [x for x in curve if x['unit'] == unit]
            assert z[-1]['cdf'] == 1 and sum(x['count'] for x in z) == r['gap']['n']
        reader_checks.append(dict(arm=name, status='pass', validation=r['validation'],
                                  complete_call_gaps=r['gap']['n'],
                                  purpose='Test-A supplementary log-reader validation only; excluded from lean pool and B'))
    for name in ('ANALYSIS.md', 'REPORT.md'):
        text = (HERE / name).read_text()
        assert text.count('<!-- TEST_A_MEASUREMENT_BEGIN -->') == text.count('<!-- TEST_A_MEASUREMENT_END -->') == 1
        section = text.split('<!-- TEST_A_MEASUREMENT_BEGIN -->')[1].split('<!-- TEST_A_MEASUREMENT_END -->')[0]
        assert all(n in section for n in SUPPLEMENT)
        assert f"{a['complete_lean_arms']}/12" in section
    result = dict(status='pass', b_cdf_groups=checked, clustered_repeat_invariance=True,
                  test_a_reader_checks=reader_checks, frozen_before=frozen, frozen_after=verify_frozen(),
                  input_files=INPUTS, script_sha256=sha(HERE / 'final_validate.py'))
    dump('final_validation.json', result)
    nominal, matched = b['sr']['all_nominal_pairs'], b['sr']['matched_only']
    cmd = ('taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src '
           'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
           '.venv/bin/python exp/offline_search/rounds/r13/astra/')
    write('final/HANDBACK.md', f"""# R13 final handback

Completed the D5 paper-support synthesis. Read [REPORT](../REPORT.md) for the short Chinese owner conclusion and technical evidence; [ANALYSIS](../ANALYSIS.md) retains all cell/dose/repeat cadence summaries and inference conventions.

The supported claim is that cost feedback sets a cadence close to periodic calling; distance changes placement within it, without a detectable pooled SR improvement at comparable budgets. B pooled D−P is {pp_ci(nominal)} pp across {nominal['pairs']:,} pairs; the frozen matched-cost subset gives {pp_ci(matched)} pp. Per-cell results remain descriptive, not selected winners.

Count correction: gates.json has 28 feedback + 28 periodic + 8 static arms, not 32 feedback arms. All observed feedback arms pass target delivery; static misses 5/8. Two paired budget mismatches remain labeled. The original GATES host/release wording and large-grid plan are superseded by coordinator D1–D5.

Paper assets: [compact table](../final_paper_table.csv), [B gap ECDF](../final_b_gap_cdf.csv), [gap summaries](../final_b_gap_summary.csv), [B budget ledger](../final_b_budget.csv), [B paired outcomes](../final_b_sr_pairs.csv), [computed B numbers and provenance](../final_b_numbers.json). For ECDF panels, facet cell/target/rep and method; `kind=any`, `unit=real_looks` matches the paper table. No-feedback static comparisons retain their actual unequal budgets.

Test-A is **{a['status']}**, with {a['complete_lean_arms']}/12 fixed lean arms locally validated at the report snapshot. The separately marked A section lists every pre-D5 supplementary arm, including the formerly unspecified GR00T L10-500 periodic .32 arm. The aborted .25 size-50 periodic arm is discarded. No extra data collection, tuning, dose choice, or method/cell selection is authorized by these reports.

Run from `/home/weiland/projects/openpi`:

```bash
{cmd}final_b_analysis.py
{cmd}final_test_a.py
{cmd}final_validate.py
```

For tomorrow's measurement update, only the second command is needed after summaries/journals/manifests/decision logs appear locally under the four `r13_test_{{groot,pi05}}_{{l10,spatial}}_rep0/runs/` roots. It reads the fixed twelve arms, flags incomplete local sync, computes pooled owner IR and complete call gaps, and fills paired SR per cell plus the fixed pooled contrast only when all paired cells are complete. Off is contextual. It replaces only the TEST_A_MEASUREMENT markers in both reports; all seven supplements are always listed descriptively. `final_test_a_numbers.json` and CSVs retain sources, hosts, times and budget-match labels. Do not use test-A to alter anything executable. Re-run the third command only if validation or this handback snapshot needs refreshing.

Validation passed: exact B cache/gate ledger reconciliation, ECDF count/mean/CDF checks, bootstrap invariance to duplicate init observations, and completed decision-log parsing on the two explicitly allowed supplementary arms. Results and hashes are in [final_validation.json](../final_validation.json). No GPU, launches, remote commands, git operations or writes outside this directory were used.

The original `../HANDBACK.md` is frozen in `../MANIFEST.sha256`, so this new `final/HANDBACK.md` is the final handback. All {frozen['members']} manifest entries still match. No frozen file was edited.
""")
    print(json.dumps(dict(status='pass', cdf_groups=checked, test_a_reader_checks=len(reader_checks), frozen=frozen)))


if __name__ == '__main__':
    main()
