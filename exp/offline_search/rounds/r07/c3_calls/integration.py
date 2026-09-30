"""Read-only C4 arm-import/manifest contract check; suppress external writes."""
import json
from unittest.mock import patch

from .common import HERE, write_json
from .package import specs


def check():
    from exp.offline_search.rounds.r07.c4_profile import prepare_profile as c4
    from exp.offline_search.rounds.r07.c4_profile import render_profile
    # C4 now renders launch recipes after writing its rows. This read-only
    # import check suppresses that output-only helper as well as C4's writer.
    with patch.object(c4.C, 'write', lambda *args: None), \
            patch.object(render_profile, 'recipes', lambda *args: None):
        profile, evaluation = c4.arm_specs(HERE / 'unused_in_memory', HERE / 'arms_eval500.json')
    for phase, rows in [('profile', profile), ('eval500', evaluation)]:
        got = {row['name']: row for row in rows}
        for row in specs(phase):
            assert got[row['name']] == row, (phase, row['name'])
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'spatial'):
            path = c4.C.HERE / 'prepared' / 'manifests' / f'{model}_{suite}_bval20.json'
            manifest = json.loads(path.read_text())
            assert manifest['role'] == 'NONTEST_BVAL' and len(manifest['selected']) == 20
            assert len({(r['task'], r['init']) for r in manifest['selected']}) == 20
    report = dict(status='PASS', C4_imported_profile_rows_equal=8,
                  C4_imported_eval500_rows_equal=8, non_test_manifests_checked=4,
                  episodes_per_profile_arm=20, external_writes=0)
    write_json(HERE / 'integration.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(check()))
