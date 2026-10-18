import json
from pathlib import Path
from unittest.mock import patch

import pytest

from . import vfix


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def _campaign(tmp_path, rows, status='PARTIAL'):
    root, source = tmp_path/'fa_ls_x', tmp_path/'src'
    _write(source/'runtime.tar', 'SOURCE_RUNTIME')
    _write(source/'RUNTIME_RELEASE.json', {'identity': 'source'})
    _write(root/'runtime.tar', 'FIXED_COPY_FROM_EARLIER_PREPARE')
    _write(root/'RUNTIME_RELEASE.json', {'identity': 'fixed'})
    _write(root/'campaign.json', dict(longseg_1017={'clock_unit': 'segment_decision'}, server_node='local', filter_failed_source=str(source),
                                      server_run=str(root/'servers')))
    _write(root/'EPISODE_PLAN.json', {'tasks': [{'task_uid': 'a'}, {'task_uid': 'b'}]})
    _write(root/'assets/cache_inputs/fit/fit.npz', 'ARRAYS')
    _write(root/'assets/cache_inputs/manifest.json', '{}')
    _write(root/'assets/ffk_bootstrap.py', 'BOOT')
    for name in ('workers.tar', 'STUDY.json', 'FFK_DEFINITION.json', 'OFFLINE_READY.json', 'PREPARED.json'):
        _write(root/name, '{}')
    _write(root/'LAUNCH.json', {'pid': 2**22+12345})
    _write(root/'RECEIPT.json', dict(status=status, all_owned_closed=True, owned_local=[{'label': 'driver'}]))
    _write(root/'driver/journal.jsonl', ''.join(json.dumps(r)+'\n' for r in rows))
    _write(root/'WORKER_DEPLOYMENT.json', {'local': dict(stage='/tmp/r17_formal/fa_ls_x/local',
        worker_run='/data/mw_finetune/qualification/local_workers/runs/fa_ls_x_local')})
    return root


def test_retry_discards_only_error_rows_and_restores_source_runtime(tmp_path):
    error = dict(task_uid='a', accepted=True, status='failed', success=False,
                 error="FatalEpisodeError: retry budget exhausted; RuntimeError('LIBERO inference failed')")
    root = _campaign(tmp_path, [error])
    archived = []
    def archive(*args, check_only=False):
        archived.append((args, check_only))
        return dict(status='READY' if check_only else 'ARCHIVED')
    with patch('exp.offline_search.rounds.r17.autoscale_1017.reset_remote.run', archive), \
            patch.object(vfix, 'repin', return_value=dict(status='REPINNED', name='fa_ls_x', bootstrap_sha256='new')), \
            patch.object(vfix, 'update_catalogs', return_value=[]):
        proof = vfix.retry(root, tmp_path/'fa_ls_x.before_retry_1435', prepare=False)
    backup = tmp_path/'fa_ls_x.before_retry_1435'
    assert (backup/'driver/journal.jsonl').read_text().count('FatalEpisodeError') == 1
    assert (backup/'RECEIPT.json').exists() and (backup/'LAUNCH.json').exists()
    assert not any((root/n).exists() for n in ('driver', 'RECEIPT.json', 'LAUNCH.json', 'PREPARED.json'))
    assert (root/'runtime.tar').read_text() == 'SOURCE_RUNTIME'
    assert json.loads((root/'RUNTIME_RELEASE.json').read_text()) == {'identity': 'source'}
    assert (root/'assets/cache_inputs/fit/fit.npz').samefile(backup/'assets/cache_inputs/fit/fit.npz')
    assert not (root/'assets/ffk_bootstrap.py').samefile(backup/'assets/ffk_bootstrap.py')
    assert proof['error_rows_discarded'] == 1 and proof['clean_accepted'] == 0
    assert [c for _, c in archived] == [True, True, False, False]
    # A second invocation resumes after the reset instead of resetting again.
    with patch.object(vfix, 'repin', side_effect=AssertionError('already repinned')):
        (root/'PREPARED.json').write_text('{}')
        assert vfix.retry(root, backup, prepare=False)['backup'] == str(backup)


def test_retry_refuses_any_clean_episode(tmp_path):
    clean = dict(task_uid='b', accepted=True, status='failed', success=False, error=None)
    root = _campaign(tmp_path, [clean])
    with pytest.raises(ValueError, match='clean accepted episodes exist'):
        vfix.retry(root, tmp_path/'fa_ls_x.before_retry', prepare=False)
    assert (root/'RECEIPT.json').exists() and not (tmp_path/'fa_ls_x.before_retry').exists()


def test_embedded_runtime_and_exact_fallback_diff():
    runtime = (vfix.KNOB/'runtime.py').read_text()
    old = runtime.replace(vfix.NEW_BLOCK, vfix.OLD_LINE, 1)
    boot = 'x=1\nffk_runtime=dict(__name__="__main__");exec(compile(%r,"ffk1015_runtime","exec"),ffk_runtime)\n' % old
    assert vfix.embedded_runtime(boot) == old
    assert old.replace(vfix.OLD_LINE, vfix.NEW_BLOCK, 1) == runtime and old != runtime
