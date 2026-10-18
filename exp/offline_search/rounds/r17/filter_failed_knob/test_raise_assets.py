import copy
import json
import shutil

import pytest

from .build import complete_cache_inputs
from .common import sha


def test_raise_uses_only_missing_a_cells_after_byte_verification(tmp_path):
    official=tmp_path/'official';target=tmp_path/'raise'
    def row(n):
        return dict(suite='libero_spatial',library_size=n,
            view_rel=f'libero_spatial/parent/subsets/size_{n}',
            fit_rel=f'libero_spatial/n{n}_C10/fit',stall_rel=f'libero_spatial/n{n}_C10/stall')
    files={}
    def put(name,data):
        path=official/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data);files[name]=sha(path)
    put('libero_spatial/parent/READY.json',b'parent')
    for n in (50,100,200):
        cell=row(n)
        for prefix,names in ((cell['view_rel'],('VIEW.json','row_indices.npy')),
                (cell['fit_rel'],('READY.json','fit.npz')),
                (cell['stall_rel'],('READY.json','stall.pkl'))):
            for name in names:put(prefix+'/'+name,f'{n}/{name}'.encode())
    (official/'manifest.json').write_text(json.dumps(dict(groups=[row(n) for n in (50,100,200)],files=files)))
    existing={n:p for n,p in files.items() if 'size_50/' in n or '/n50_' in n or n=='libero_spatial/parent/READY.json'}
    for name in existing:
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(official/name,path)
    inputs=dict(files=copy.deepcopy(existing))
    # Existing-only raises remain byte-identical, as required for queued runs.
    before=copy.deepcopy(inputs)
    complete_cache_inputs(target,official,[row(50)],inputs)
    assert inputs==before and {str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}==existing
    shared=official/row(50)['fit_rel']/'fit.npz';original=shared.read_bytes();shared.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache input differs'):
        complete_cache_inputs(target,official,[row(100)],inputs)
    assert inputs==before and not (target/row(100)['view_rel']).exists()
    shared.write_bytes(original)
    with pytest.raises(ValueError,match='lacks raised cell'):
        complete_cache_inputs(target,official,[row(300)],inputs)
    assert inputs==before
    complete_cache_inputs(target,official,[row(50),row(100)],inputs)
    assert inputs['files']=={n:p for n,p in files.items() if 'size_200/' not in n and '/n200_' not in n}
    assert all(sha(target/n)==sha(official/n)==p for n,p in inputs['files'].items())
    assert not (target/row(200)['fit_rel']).exists()
