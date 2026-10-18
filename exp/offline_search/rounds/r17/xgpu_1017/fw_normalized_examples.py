"""CPU audit of selected worst-wire cases. These are examples, not global maxima."""
import itertools
import json
from pathlib import Path
import numpy as np
from artifacts import load_tree, sha


def audit(paths, keys, stats):
    tensors = next(iter(stats.values()))['tensors']
    amin = np.asarray(tensors['action.min'], dtype=np.float64)[:6]
    gain = (np.asarray(tensors['action.max'], dtype=np.float64)[:6]-amin)/2
    samples = {tuple(key): [] for key in keys}
    for root in map(Path, paths):
        run = json.loads((root/'RUN.json').read_text())
        for directory in run['artifacts']:
            rows = json.loads((root/directory/'ROWS.json').read_text())
            for entry in rows['rows']:
                key = tuple(entry['key'])
                if key not in samples:
                    continue
                path=root/directory/entry['file']
                assert sha(path.with_suffix('.json'))==entry['sha256']
                value=load_tree(path)
                samples[key].append((str(root.name), rows['round'], value['response']['actions']))
    output=[]
    for key,values in samples.items():
        assert len(values)==4
        result=dict(key=key, repeated=4, shape=list(values[0][2].shape),
                    wire_max_abs=0., normalized6_max_abs=0., normalized6_first5_max_abs=0.)
        for (_,_,a),(_,_,b) in itertools.combinations(values,2):
            a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
            delta=np.abs((a[...,:6]-b[...,:6])/gain)
            result['wire_max_abs']=max(result['wire_max_abs'],float(np.abs(a-b).max()))
            result['normalized6_max_abs']=max(result['normalized6_max_abs'],float(delta.max()))
            result['normalized6_first5_max_abs']=max(result['normalized6_first5_max_abs'],float(delta[:5].max()))
        output.append(result)
    return dict(scope='selected worst-wire examples; inverse-affine six motion channels only; no global normalized maximum claim',
                action_min6=amin.tolist(),action_gain6=gain.tolist(),examples=output)
