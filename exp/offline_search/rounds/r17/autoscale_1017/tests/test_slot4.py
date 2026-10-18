"""Slot4 failures through real closure/reset/collector code; no remote calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import kernel_evidence as k, reset_remote
from exp.r17_integration.ops import library_salvage as salvage


def test_real_large_receipt_closure_uses_node_files_and_bounded_argv(tmp_path):
    source=Path('/data/mw_finetune/eval/fa_ls_gr_r_p_m/RECEIPT.json')
    receipt=json.loads(source.read_text())
    receipt=copy.deepcopy(receipt)
    for i,e in enumerate(receipt['owned_local']):e.update(pid=1000000000+i,pgid=1000000000+i,start_ticks='1')
    expected={};calls=[]
    for i,owned in enumerate(receipt['owned_remote']):
        path=tmp_path/('receipt_'+str(i)+'.json')
        for j,role in enumerate(('child','supervisor')):
            if role in owned['receipt']:
                e=owned['receipt'][role];e.update(pid=1000000100+2*i+j,pgid=1000000100+2*i+j,start_ticks='1')
                expected.setdefault(owned['node'],[]).append(e['pid'])
        owned['request']['receipt']=str(path)
        owned['receipt']['request']['receipt']=str(path)
        path.write_text(json.dumps(owned['receipt']))
    def remote(node,argv):
        # Old real code exceeds Linux MAX_ARG_STRLEN and raises errno 7 here.
        out=subprocess.run([sys.executable,'-c',argv[-1]],capture_output=True,text=True,check=True)
        assert max(map(len,argv)) < 32768
        assert 'read_text' in argv[-1] and str(tmp_path) in argv[-1]
        calls.append(node)
        return out.stdout
    result=salvage.prove_closed(receipt,{'run_name':'CPU_SLOT4_CLOSURE'},remote)
    assert result['all_owned_closed'] and set(calls)==set(expected)
    assert {n:[e['pid'] for e in rows] for n,rows in result['processes']['remote'].items()}==expected


def reset_fixture(tmp_path):
    root=tmp_path/'fa_cpu';root.mkdir();(root/'mps_control').mkdir()
    proc=tmp_path/'proc';p=proc/'3514';p.mkdir(parents=True)
    (p/'cmdline').write_bytes(b'python\0unrelated\0')
    (p/'cwd').symlink_to(tmp_path,target_is_directory=True)
    return root,proc,p


@pytest.mark.parametrize('field',['cwd','cmdline','stat'])
def test_reset_skips_uninspectable_unrelated_process(tmp_path,monkeypatch,field):
    root,proc,p=reset_fixture(tmp_path)
    method={'cwd':'resolve','cmdline':'read_bytes','stat':'stat'}[field]
    original=getattr(Path,method)
    def denied(path,*a,**kw):
        if path==(p if field=='stat' else p/field):raise PermissionError(13,'denied',str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,method,denied)
    result=reset_remote.archive(dict(root=str(root),backup=str(tmp_path/'fa_cpu.failed')),
                                campaign_parent=tmp_path,proc_root=proc)
    assert result['status']=='ARCHIVED' and (tmp_path/'fa_cpu.failed/mps_control').is_dir()


@pytest.mark.parametrize('unreadable',['cwd','cmdline'])
def test_reset_still_refuses_run_proven_by_other_readable_field(tmp_path,monkeypatch,unreadable):
    root,proc,p=reset_fixture(tmp_path)
    if unreadable=='cwd':(p/'cmdline').write_bytes(b'python\0'+str(root/'servers').encode()+b'\0')
    else:
        (p/'cwd').rename(p/'old_cwd');(p/'cwd').symlink_to(root,target_is_directory=True)
    method='resolve' if unreadable=='cwd' else 'read_bytes';original=getattr(Path,method)
    def denied(path,*a,**kw):
        if path==p/unreadable:raise PermissionError(13,'denied',str(path))
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,method,denied)
    with pytest.raises(ValueError,match='still references|cwd is inside'):
        reset_remote.archive(dict(root=str(root),backup=str(tmp_path/'fa_cpu.failed')),
                             campaign_parent=tmp_path,proc_root=proc)
    assert root.is_dir() and not (tmp_path/'fa_cpu.failed').exists()


@pytest.mark.parametrize('in_window',[False,True])
def test_plain_dmesg_covers_old_acceptance_excludes_later_oom(tmp_path,monkeypatch,in_window):
    original=Path.read_text
    def read(path,*a,**kw):
        if str(path)=='/proc/uptime':return '300.00 1.0\n'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    calls=[]
    def execute(argv,**kwargs):
        calls.append(argv)
        if argv==['dmesg']:
            output='[ 50.000000] retained log\n[ 90.000000] Out of memory: Killed process 123\n[ 250.000000] Out of memory: Killed process 456\n'
            if in_window:output+='[ 150.000000] Out of memory: Killed process 789\n'
            return SimpleNamespace(returncode=0,stdout=output,stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr='dmesg: unknown time format: raw')
    result=k.collect(tmp_path,100,ended_at=200,now=lambda:300,execute=execute,
                     snapshot={'time':300,'boot_epoch':0,'vmstat':{'oom_kill':45}},persist=False)
    assert result['source']=='dmesg' and result['coverage']
    assert result['status']==('FAIL' if in_window else 'PASS')
    assert [x['time'] for x in result['oom_lines']]==([150] if in_window else [])
    assert calls[0]==['dmesg']


def test_receipt_file_identity_mismatch_cannot_prove_closed(tmp_path):
    path=tmp_path/'owner.json';path.write_text(json.dumps(dict(child=dict(pid=12,start_ticks='changed',pgid=12))))
    with pytest.raises(ValueError,match='identity changed'):
        salvage.receipt_processes([dict(path=str(path),identities={'child':dict(pid=12,start_ticks='original',pgid=12)})])


def test_many_receipt_files_are_batched_below_argument_limit(tmp_path):
    refs=[]
    for i in range(150):
        entry=dict(pid=1000000000+i,start_ticks='1')
        path=tmp_path/('process_'+str(i)+'.json');record=dict(child=entry,request={'receipt':str(path)})
        path.write_text(json.dumps(record));refs.append(dict(node='h100',receipt=record))
    calls=[]
    def remote(node,argv):
        assert len(argv[-1].encode()) < 32768
        calls.append(argv)
        return subprocess.run([sys.executable,'-c',argv[-1]],capture_output=True,text=True,check=True).stdout
    result=salvage.prove_closed(dict(owned_local=[dict(pid=1000000200,start_ticks='1')],owned_remote=refs),
                                {'run_name':'CPU_SLOT4_BATCH'},remote)
    assert len(calls)>1 and len(result['processes']['remote']['h100'])==150


def test_readable_but_rotated_dmesg_retains_source_and_bounds(tmp_path,monkeypatch):
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw:'300 0' if str(p)=='/proc/uptime' else original(p,*a,**kw))
    def execute(argv,**kwargs):
        return SimpleNamespace(returncode=0,stdout='[ 250.0] Out of memory: Killed process 123\n',stderr='') if argv==['dmesg'] else SimpleNamespace(returncode=1,stdout='',stderr='permission denied')
    result=k.collect(tmp_path,100,ended_at=200,now=lambda:300,execute=execute,
                     snapshot={'time':300,'boot_epoch':0,'vmstat':{'oom_kill':45}},persist=False)
    assert result['status']=='UNKNOWN' and result['source']=='dmesg'
    assert not result['coverage'] and not result['oom_lines']
    assert result['log_evidence'][0]['retained_from']==250


def test_closure_refuses_known_group_even_when_cmdline_unreadable(tmp_path,monkeypatch):
    proc=tmp_path/'proc';p=proc/'777';p.mkdir(parents=True)
    (p/'stat').write_text('777 (child) S 0 777 '+'0 '*20)
    original=Path.read_bytes
    def read(path,*a,**kw):
        if path==p/'cmdline':raise PermissionError('unreadable argv')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_bytes',read)
    monkeypatch.setattr(salvage,'Path',lambda name:proc if str(name)=='/proc' else Path(name))
    with pytest.raises(ValueError,match='group member still alive'):
        salvage.process_probe([dict(pid=1000000000,start_ticks='1',pgid=777)],'CPU_GROUP')
