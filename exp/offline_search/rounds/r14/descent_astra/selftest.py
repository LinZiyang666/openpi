"""Synthetic-only tests. An audit hook forbids all real trace-root reads."""
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

import analyze as a


def fence(event,args):
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to('/home/weiland/trace_runs'):
            raise PermissionError('Synthetic tests may not read trace roots')
        write=bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write and not path.is_relative_to(a.HERE):
            raise PermissionError('Selftest write outside descent_astra')


sys.addaudithook(fence)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def fake_root(directory,p,selection,original_caps):
    """One complete synthetic B cell, using real selection metadata, no outcomes."""
    cell='pi05_l10_50'
    root=directory/'synthetic_pi05_l10_50'
    caps=copy.deepcopy(original_caps)
    selected=selection['suites']['l10']
    manifest=dict(dev=True,init_pool='B',pairs=selected['pairs'],
        selection_sha256=p['input_sha256'][str((a.R14/'descent/selection.json').relative_to(a.REPO))])
    dump(root/'dev_selection.json',manifest)
    fit=directory/'synthetic_fit.bin';fit.write_bytes(b'synthetic fit; never executable')
    digest=a.sha(fit.read_bytes())
    contracts={};arms=[]
    for level in a.ARMS:
        name=a.arm_name(cell,level)
        caps['cells'][cell]['fit_sha256_by_level'][level]=digest
        row=dict(arm=name,dev=True,init_pool='B',episodes=300,model='pi05',r14_descent_level=level,
                 plugin_args=['--os-fit-artifact',str(fit)],manifest=str(root/'dev_selection.json'))
        arms.append(row)
        contract=dict(dev=True,init_pool='B',arm=name,model='pi05',
                      manifest_sha256=a.sha(json.dumps(sorted(map(tuple,selected['pairs'])),separators=(',',':')).encode()),
                      fit_sha256=digest,subset=selected['subsets']['pi05'])
        contracts[name]=contract
        run=root/'runs'/name
        dump(run/'pool_contract.json',contract)
        dump(run/'manifest.json',manifest)
        dump(run/'selection.json',dict(init_pool='B',manifest=str(run/'manifest.json')))
        journal=[];decs=[]
        for t,i in selected['pairs']:
            uid=f'{name}:libero_10:{t}:{i}'
            journal.append(dict(task_uid=uid,init_pool='B',accepted=True,status='done',error=None,success=True,attempt=2))
            decs.append(dict(ev='dec',uid=uid,step=0,attempt=1,hit=True,vision=True,miss_k=None))
            decs.append(dict(ev='dec',uid=uid,step=0,attempt=2,hit=False,vision=True,miss_k=10,ok=True))
        # A normal failure is still complete, an error attempt must not replace it.
        journal[0].update(success=False,status='failed')
        journal.append(dict(journal[0],error='synthetic rejected retry',attempt=3,success=True))
        decs.append(dict(decs[1]))
        jsonl(run/'client/journal.jsonl',journal)
        jsonl(run/'server_synthetic/decisions_0.jsonl',decs)
        dump(run/'summary.json',dict(init_pool='B',arm=name,complete=300,success=299,
            per_task={str(t):dict(n=30,success=29 if t==0 else 30) for t in range(10)},
            cost_ledger=dict(full_stage_costs=dict(s1=.152,s2=.410,s3=.438,k=10),
                             decisions=300,vision_decisions=300,misses=300,controls=1500,ir_per_five_controls=1.)))
    dump(root/'arms.json',arms);dump(root/'dev_contracts.json',contracts)
    dump(root/'ROOT.json',dict(init_pool='B',cell=cell,episodes_per_arm=300,arm_count=8,
        definitions_sha256={name:a.sha((root/name).read_bytes()) for name in ('arms.json','dev_contracts.json','dev_selection.json')},
        fit_sha256={str(fit):digest}))
    return root,cell,caps


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader=a.Reader();cls.p=a.verify_protocol(reader)
        cls.selection=reader.json(a.R14/'descent/selection.json')
        cls.caps=reader.json(a.R14/'descent/cap_tables.json')
        cls.fits=reader.json(a.R14/'astra/allocation_fits.json')
        cls.csv=list(csv.DictReader(io.StringIO(reader.read(a.R14/'astra/allocation_tasks.csv').decode())))
        (a.HERE/'_selftest').mkdir(exist_ok=True)

    def test_quantization_and_boundary(self):
        pure=np.ones(30,dtype=int)
        for losses in range(4):
            y=pure.copy();y[:losses]=0
            stats={l:a.paired(y,pure) for l in a.ORDER}
            stats['c10']=a.paired(pure,pure,identical=True)
            for tol in (0,2,5):
                selected=a.descend(stats,tol)['level']
                self.assertEqual(selected,'off' if 100*losses<=tol*30 else 'c10')
        row=dict(n=300,successes=285,pure_successes=300,sr=.95,simultaneous_lower95=-1.)
        stats={l:row for l in a.ORDER};stats['c10']=dict(row,successes=300,sr=1.)
        self.assertEqual(a.descend(stats,5)['level'],'off')

    def test_first_failure_no_reentry(self):
        pure=np.ones(30,dtype=int)
        stats={l:a.paired(pure,pure,identical=l=='c10') for l in a.ORDER}
        fail=pure.copy();fail[:2]=0
        stats['ir40']=a.paired(fail,pure)
        d=a.descend(stats,5)
        self.assertEqual(d['level'],'ir45')
        self.assertEqual(d['first_failure'],'ir40')
        self.assertEqual(d['passing_below_first_failure'],['ir32','ir25','off'])

    def test_improvements_and_offsetting_discordance(self):
        pure=np.array([0,1]+[1]*28)
        y=np.array([1,0]+[1]*28)
        s=a.paired(y,pure)
        self.assertEqual((s['gains'],s['losses'],s['delta']),(1,1,0.))
        stats={l:s for l in a.ORDER}
        self.assertEqual(a.descend(stats,0)['level'],'off')

    def test_confidence_not_degenerate_at_no_discordance(self):
        y=np.ones(30,dtype=int);s=a.paired(y,y)
        self.assertAlmostEqual(s['lower95'],-(1-.025**(1/30)))
        self.assertLess(s['simultaneous_lower95'],s['lower95'])
        self.assertEqual(a.paired(y,y,identical=True)['ci95'],[0,0])
        stats={l:s for l in a.ORDER};stats['c10']=a.paired(y,y,identical=True)
        self.assertEqual(a.descend(stats,5,True)['level'],'c10')
        pooled=a.cell_pair(np.tile(y,(10,1)),np.tile(y,(10,1)))
        self.assertLess(pooled['lower95'],s['lower95'])

    def test_cap_mapping_and_rank_ties(self):
        caps=self.caps['cells']['pi05_l10_50']['periodic_cap_by_level']
        for l in a.ORDER[:-1]:
            self.assertEqual(a.nearest_level(caps[l],caps),l)
        midpoint=math.expm1((math.log1p(caps['ir45'])+math.log1p(caps['ir40']))/2)
        self.assertEqual(a.nearest_level(midpoint,caps),'ir45')
        self.assertEqual(a.nearest_level(1e9,caps),'ir25')
        self.assertIsNone(a.ranks(range(10),[1]*10)['spearman'])
        self.assertAlmostEqual(a.ranks([0,0,1,1],[0,0,1,1])['kendall_tau_b'],1.)

    def test_accepted_retry_error_pool_and_missing(self):
        good=dict(task_uid='x:suite:0:1',init_pool='B',accepted=True,status='done',success=False,attempt=1)
        later=dict(good,success=True,attempt=2)
        rows,errors=a.accepted_rows([good,later,dict(good,error='error',attempt=3)],{(0,1)})
        self.assertEqual(rows[good['task_uid']]['attempt'],2);self.assertEqual(errors,1)
        with self.assertRaises(ValueError):a.accepted_rows([dict(good,init_pool='A')],{(0,1)})
        with self.assertRaises(ValueError):a.accepted_rows([good],{(0,1),(0,2)})

    def test_loader_on_synthetic_logs(self):
        with tempfile.TemporaryDirectory(dir=a.HERE/'_selftest') as temp:
            root,cell,caps=fake_root(Path(temp),self.p,self.selection,self.caps)
            reader=a.Reader()
            arms=a.load_cell(root,cell,self.p,self.selection,caps,reader)
            reader.stable()
            self.assertEqual(len(arms),8)
            self.assertEqual(arms['off'].stats()['successes'],299)
            self.assertEqual(arms['off'].audit['rejected_attempt_decisions'],300)
            self.assertEqual(arms['off'].audit['duplicate_decisions'],1)
            self.assertEqual(arms['off'].stats()['ir'],1.)
            path=root/'runs'/a.arm_name(cell,'off')/'summary.json'
            summary=json.loads(path.read_text());summary['cost_ledger']['misses']=299;dump(path,summary)
            with self.assertRaisesRegex(ValueError,'counts differ'):
                a.load_cell(root,cell,self.p,self.selection,caps,a.Reader())

    def test_changed_input_rejected(self):
        with tempfile.TemporaryDirectory(dir=a.HERE/'_selftest') as temp:
            path=Path(temp)/'source';path.write_text('before')
            r=a.Reader();r.read(path);path.write_text('after')
            with self.assertRaises(ValueError):r.stable()

    def test_loader_contract_and_run_manifest_bindings(self):
        with tempfile.TemporaryDirectory(dir=a.HERE/'_selftest') as temp:
            root,cell,caps=fake_root(Path(temp),self.p,self.selection,self.caps)
            name=a.arm_name(cell,'c10');run=root/'runs'/name
            selected=json.loads((run/'selection.json').read_text())
            for bad in (dict(selected,init_pool='A'),dict(selected,manifest=str(root/'dev_selection.json'))):
                dump(run/'selection.json',bad)
                with self.assertRaisesRegex(ValueError,'Wrong run selection binding'):
                    a.load_cell(root,cell,self.p,self.selection,caps,a.Reader())
            dump(run/'selection.json',selected)
            original=(run/'manifest.json').read_bytes()
            (run/'manifest.json').write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'Run manifest differs'):
                a.load_cell(root,cell,self.p,self.selection,caps,a.Reader())
            (run/'manifest.json').write_bytes(original)
            contracts=json.loads((root/'dev_contracts.json').read_text())
            contracts[name]['manifest_sha256']=a.sha((root/'dev_selection.json').read_bytes())
            dump(root/'dev_contracts.json',contracts)
            dump(run/'pool_contract.json',contracts[name])
            seal=json.loads((root/'ROOT.json').read_text())
            seal['definitions_sha256']['dev_contracts.json']=a.sha((root/'dev_contracts.json').read_bytes())
            dump(root/'ROOT.json',seal)
            with self.assertRaisesRegex(ValueError,'Wrong contract manifest'):
                a.load_cell(root,cell,self.p,self.selection,caps,a.Reader())

    def synthetic_arms(self):
        arms={}
        for level,ir,n in [('c10',1.,4),('ir45',.6,8),('ir40',.5,5),('ir32',.3,3),('ir25',.2,6),('off',.1,2),
                           ('task_ir32',.4,4),('task_ir40',.55,6)]:
            y=np.ones((10,30),dtype=int)
            if level=='ir32':y[0,:2]=0
            if level=='ir25':y[1,:3]=0
            counts=np.full((10,30),n,dtype=int)
            arms[level]=a.Episodes(y,counts,counts.copy(),counts.copy(),counts*ir,dict(synthetic=True,summary_stage_ir=ir))
        return arms

    def test_exact_ir_mixture_ratio_of_sums_and_no_extrapolation(self):
        arms=self.synthetic_arms();idx=a.bootstrap_indices(self.p)
        m=a.matched_ir(arms,'ir32',self.p,idx)
        mix=m['exact_ir_expected_mixture']
        self.assertEqual((mix['low'],mix['high']),('ir32','ir40'))
        self.assertAlmostEqual(mix['high_episode_weight'],.375)
        self.assertAlmostEqual(mix['ir'],.4)
        arms['task_ir32'].cost=arms['task_ir32'].n*.8
        self.assertFalse(a.matched_ir(arms,'ir32',self.p,idx)['exact_ir_expected_mixture']['available'])

    def test_full_analysis_outputs_repeatable(self):
        all_arms={c:self.synthetic_arms() for c in self.p['cells']}
        result=a.analyze_cells(all_arms,self.p,self.caps,self.fits,self.csv)
        files=a.outputs(result,self.p)
        second=a.outputs(a.analyze_cells(all_arms,self.p,self.caps,self.fits,self.csv),self.p)
        self.assertEqual(json.dumps(files,sort_keys=True),json.dumps(second,sort_keys=True))
        self.assertEqual(files['levels.json'],files['levels_tol5pp.json'])
        self.assertEqual(files['levels_tol0pp.json'],files['levels_tol2pp.json'])
        self.assertEqual(files['uniform.json'],files['uniform_tol5pp.json'])
        self.assertTrue(files['DESCENT.md'].startswith('# 给负责人的结果说明'))
        for cell in self.p['cells']:
            self.assertEqual(files['levels.json'][cell]['0'],'ir40')
            self.assertEqual(files['levels.json'][cell]['1'],'ir32')
            self.assertEqual(files['levels.json'][cell]['2'],'off')
        with tempfile.TemporaryDirectory(dir=a.HERE/'_selftest') as temp:
            a.write_outputs(files,Path(temp))
            self.assertEqual(json.loads((Path(temp)/'levels.json').read_text()),files['levels.json'])

    def test_write_boundary(self):
        with self.assertRaises(ValueError):a.write_outputs({'levels.json':{}},a.R14/'descent')
        with self.assertRaises(ValueError):a.write_outputs({'protocol.json':{}},a.HERE)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(synthetic_only=True,trace_reads_forbidden=True,tests_run=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful())
    (a.HERE/'selftest_result.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
