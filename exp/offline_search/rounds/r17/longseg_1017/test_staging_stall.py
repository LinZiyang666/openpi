import os

from .staging import metadata_tree


def test_metadata_tree_copies_stall_artifacts_without_touching_shared_inodes(tmp_path):
    shared = tmp_path/'shared'/'q3'/'abc'
    shared.mkdir(parents=True)
    for name in ('READY.json', 'stall.json', 'stall.pkl'):
        (shared/name).write_text(name)
    (tmp_path/'shared'/'model_spec.json').write_text('{}')
    run = tmp_path/'run'
    (run/'assets'/'q3').mkdir(parents=True)
    os.link(tmp_path/'shared'/'model_spec.json', run/'assets'/'model_spec.json')
    (run/'assets'/'q3'/'abc').mkdir()
    for name in ('READY.json', 'stall.json', 'stall.pkl'):
        os.link(shared/name, run/'assets'/'q3'/'abc'/name)
    before = {n: (os.stat(shared/n).st_nlink, os.stat(shared/n).st_ctime_ns) for n in ('READY.json', 'stall.pkl')}
    dest = metadata_tree(run)
    after = {n: (os.stat(shared/n).st_nlink, os.stat(shared/n).st_ctime_ns) for n in ('READY.json', 'stall.pkl')}
    assert before == after  # running servers' stall stamps stay valid
    assert (dest/'q3'/'abc'/'stall.pkl').read_text() == 'stall.pkl'
    assert not (dest/'q3'/'abc'/'stall.pkl').samefile(shared/'stall.pkl')
    assert (dest/'model_spec.json').samefile(tmp_path/'shared'/'model_spec.json')  # other inputs still linked
    metadata_tree(run)  # re-stage keeps identical copies
    assert not list((dest/'q3'/'abc').glob('*.previous_*'))
