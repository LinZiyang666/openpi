"""CPU-only launcher host mappings for the pi05/GR00T model islands."""
import pytest

from exp.r17_integration.ops.formal_eval import CONTROL_PYTHON, host_python, pythonpath, server_env

LOCAL_GROOT = '/home/weiland/projects/openpi_ext/third_party/gr00t_n15'
REMOTE_ROOT = '/data/mw_finetune/qualification/formal_runtime_test'


@pytest.mark.parametrize('model,node,expected', [
    ('pi05', 'local', CONTROL_PYTHON),
    ('pi05', 'h100', '/home/exouser/openpi/.venv/bin/python'),
    ('groot', 'local', '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python'),
    ('groot', 'h100', '/home/exouser/gr00t_n15_venv/.venv/bin/python'),
    ('dp', 'h100', '/data/mw_finetune/dp/env/bin/python'),
    ('fastwam', 'h100', '/data/mw_finetune/envs/fastwam/bin/python'),
    ('cosmos', 'h100', '/data/cosmos/cosmos-policy/.venv/bin/python'),
])
def test_interpreter_mapping(model, node, expected):
    campaign = dict(model=model, server_node=node,
                    vendor='/data/cosmos/cosmos-policy' if model == 'cosmos' else LOCAL_GROOT)
    assert host_python(campaign) == expected
    if model == 'groot':
        assert campaign['vendor'] == ('/home/exouser/gr00t_n15' if node == 'h100' else LOCAL_GROOT)


@pytest.mark.parametrize('model', ['pi05', 'groot'])
def test_h100_native_runtime_and_caches(model):
    # server_env also works before host_python normalizes the campaign vendor.
    campaign = dict(model=model, server_node='h100', vendor=LOCAL_GROOT)
    env = server_env(campaign, REMOTE_ROOT)
    paths = env['PYTHONPATH'].split(':')
    assert paths[:3] == pythonpath(REMOTE_ROOT).split(':')
    assert not any('/home/weiland/' in p or 'python_deps312' in p for p in paths)
    assert env['HF_HOME'] == '/home/exouser/.cache/huggingface'
    if model == 'pi05':
        assert env['OPENPI_DATA_HOME'] == '/home/exouser/.cache/openpi'
        assert paths == pythonpath(REMOTE_ROOT).split(':')
    else:
        assert paths[-1] == '/home/exouser/gr00t_n15'
        assert env['NO_ALBUMENTATIONS_UPDATE'] == '1'


def test_local_and_other_islands_keep_existing_paths():
    assert host_python(dict(model='pi05')) == CONTROL_PYTHON
    env = server_env(dict(model='groot', server_node='local', vendor=LOCAL_GROOT), '/local/runtime')
    assert env['PYTHONPATH'] == pythonpath('/local/runtime', LOCAL_GROOT)
    assert 'HF_HOME' not in env and 'OPENPI_DATA_HOME' not in env
    env = server_env(dict(model='dp', server_node='h100'), REMOTE_ROOT)
    assert env['PYTHONPATH'].endswith('/data/mw_finetune/qualification/python_deps312')
