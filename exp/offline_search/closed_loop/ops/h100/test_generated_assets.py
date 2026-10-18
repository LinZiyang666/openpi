"""Two campaigns may relocate the same fit without sharing a mutable mirror."""
import copy
import json
from pathlib import Path
import pickle

from exp.offline_search.closed_loop.ops.h100 import assets, node
from exp.offline_search.closed_loop.ops.h100.test_h100 import fixture_run


def test_rewritten_external_assets_are_immutable_and_inputs_unchanged(tmp_path, monkeypatch):
    root, store, base, row = fixture_run(tmp_path)
    fit = tmp_path / 'shared.pkl'
    blob = pickle.loads((root / 'fits/sample.pkl').read_bytes())
    blob.update(kwargs={}, run='<RUN>/runs/sample')
    fit.write_bytes(pickle.dumps(blob, protocol=4))
    row['kwargs'] = {}
    row['plugin_args'][-1] = str(fit)
    (root / 'arms.json').write_text(json.dumps([row]))
    other = root.with_name('other')
    (other / 'config').mkdir(parents=True)
    other_row = copy.deepcopy(row)
    for field in ('yaml', 'matrix'):
        original = Path(row[field])
        target = other / 'config' / original.name
        target.write_bytes(original.read_bytes())
        other_row[field] = str(target)
    (other / 'arms.json').write_text(json.dumps([other_row]))
    source = fit.read_bytes()
    mapper = assets.remap

    def remap(value, run, s=store, b=base):
        if value == str(fit):
            return str(base / 'mirror/shared.pkl')
        if value == str(tmp_path / 'native.pkl'):
            return str(base / 'mirror/native.pkl')
        return mapper(value, run, s, b)

    monkeypatch.setattr(assets, 'remap', remap)
    stale = base / 'mirror/shared.pkl'
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b'old closed campaign evidence')
    addresses = []
    for run in (root, other, root):
        old = assets.build_plan(run, ['sample'], tmp_path / ('old_' + run.name),
                                store=store, base=base, isolate_generated=False)
        new = assets.build_plan(run, ['sample'], tmp_path / ('new_' + run.name),
                                store=store, base=base, isolate_generated=True)
        before = next(f for f in old['files'] if f['original'] == str(fit))
        after = next(f for f in new['files'] if f['original'] == str(fit))
        assert after['sha256'] == before['sha256']
        assert Path(after['source']).read_bytes() == Path(before['source']).read_bytes()
        assert after['rel'] == assets.generated_rel('mirror/shared.pkl', before['sha256'])
        assert assets.flag(new['arms'][0]['plugin_args'], '--os-fit-artifact') == str(base / after['rel'])
        assert [(f['rel'], f['sha256']) for f in old['files'] if f['original'] != str(fit)] == [
            (f['rel'], f['sha256']) for f in new['files'] if f['original'] != str(fit)]
        expected = copy.deepcopy(old['arms'])
        expected[0]['plugin_args'][-1] = str(base / after['rel'])
        assert new['arms'] == expected
        assert not any(f['rel'] == 'mirror/shared.pkl' for f in new['files'])
        addresses.append(after['rel'])
    assert addresses[0] != addresses[1] and addresses[0] == addresses[2]
    assert stale.read_bytes() == b'old closed campaign evidence' and fit.read_bytes() == source
    assert node.sha(stale) != node.sha(fit)
