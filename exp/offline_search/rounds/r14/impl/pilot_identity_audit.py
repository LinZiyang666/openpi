"""Frozen consumed-B identities and both models' middle-size exclusions."""
import json
from pathlib import Path
from exp.offline_search.closed_loop import devset
from .common import *

def main():
    records=[];inputs={}
    for sel in json.loads((ASTRA/'pilot_selection.json').read_text())['cells']:
        cell=sel['cell'];model,suite,size=cell.split('_');consumed=set()
        for source in sel['sources']:
            path=Path(source['path'])
            if not path.is_relative_to(TRACE) or path.relative_to(TRACE).parts[0] not in ('r11_devknob_50','r12_devlooks_200'):
                raise ValueError('only frozen pre-R13 B sources are eligible')
            assert sha(path)==source['sha256'];inputs[str(path)]=sha(path)
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if 'success' in row and not row.get('error') and row.get('accepted',True):
                    assert row['init_pool']=='B'
                    consumed.add(tuple(map(int,row['task_uid'].split(':')[-2:])))
        pairs=set(map(tuple,sel['pairs']));assert len(pairs)==50 and pairs<=consumed
        actual=devset.r10_subset(model,suite,int(size))
        assert actual['excluded_inits_by_task']==sel['excluded_inits_by_task']
        assert all(i not in actual['excluded_inits_by_task'][str(t)] for t,i in pairs)
        cross=False
        if cell=='groot_l10_200':
            other=devset.r10_subset('pi05','l10',200)
            assert all(i not in other['excluded_inits_by_task'][str(t)] for t,i in pairs)
            cross=True
        records.append(dict(cell=cell,pairs=50,already_consumed=True,actual_served_subset_disjoint=True,
                            cross_model_middle_exclusion_verified=cross))
        print('IDENTITIES_EXACT',cell,flush=True)
    write_json(HERE/'pilot_identity_audit.json',dict(PASS=True,records=records,inputs_sha256=inputs,
        no_reserved_R13_selection_read=True,no_new_selection=True))

if __name__=='__main__':main()
