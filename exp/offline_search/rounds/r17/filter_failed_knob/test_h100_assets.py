"""H100 must resolve both pinned recipe paths and load_cache_config paths."""
from types import SimpleNamespace

from exp.offline_search.rounds.r17.filter_failed import stage, reset
from exp.r17_integration.ops import sync_back
from .common import dump, read


def test_h100_short_asset_aliases_and_incomplete_prepare_reset(tmp_path, monkeypatch):
    root = tmp_path/('isolated_'+tmp_path.name)
    assets = root/'assets'
    assets.mkdir(parents=True)
    dump(assets/'manifest.json', {})
    frozen = b'recipe:\n  fit_path: /original/assets/cache_inputs/fit.npz\n'
    (assets/'full.yaml').write_bytes(frozen)
    campaign = dict(server_node='h100', run_name=root.name,
                    assets='/original/assets', local_assets=str(assets), runtime='/frozen/runtime')
    installs, transfers = [], []
    def remote(node, argv, **kwargs):
        assert node == 'h100'
        if argv[0] == 'env':exec(argv[-1], {})
    def install(archive, destination, **kwargs):
        installs.append((archive, destination, kwargs['expected_sha256']))
    monkeypatch.setattr(sync_back, 'install_snapshot', install)
    launch = SimpleNamespace(remote=remote, transfer=lambda argv: transfers.append(argv),
                             pythonpath=lambda _: '/frozen', host_python=lambda _: '/sealed/python')
    stage.stage_server_inputs(launch, campaign, root)
    assert [i[1] for i in installs] == ['/original/assets', str(assets)]
    assert installs[0][0] == installs[1][0] and installs[0][2] == installs[1][2]
    assert all(argv[:2] == ['rsync', 'push'] for argv in transfers)
    assert (assets/'full.yaml').read_bytes() == frozen
    assert stage.server_asset_destinations(dict(assets='/same', local_assets='/same')) == ['/same']
    dump(root/'WORKER_DEPLOYMENT.json', {'partial': {}})
    dump(root/'PREPARED.json', {'status': 'partial'})
    dump(root/'campaign.json', campaign)
    reset.reset(root, unstage_only=True)
    record = read(next(root.glob('RESET.failed_*.json')))
    assert record['status'] == 'UNSTAGED' and record['external_renames'] == []
    for name in ('PREPARED.json', 'WORKER_DEPLOYMENT.json'):
        assert not (root/name).exists() and (root/record['renamed'][name]).exists()
    assert (assets/'full.yaml').read_bytes() == frozen
