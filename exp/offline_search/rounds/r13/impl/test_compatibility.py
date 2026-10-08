"""Diff-check actual served bytes against the saved pre-edit plugin/recorder."""
import importlib.util
import json
from pathlib import Path
import pickle

import pytest
import numpy as np

from exp.offline_search.closed_loop import plugin, look_recording
from exp.offline_search.rounds.r11.knob.recipe import METHODS
from . import mechanics
from .mechanics import EVIDENCE, Harness, load
from .prepare import HERE, R11


def previous(name):
    spec=importlib.util.spec_from_file_location(f'exp.offline_search.closed_loop.r13_previous_{name}',HERE/'before'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('model',['pi05','groot'])
def test_all_existing_R11_methods_actions_and_decisions_unchanged(model,tmp_path,monkeypatch):
    old=previous('plugin');old._git_head=lambda:None
    monkeypatch.setattr(plugin,'_git_head',lambda:None)
    cell=f'{model}_l10_50';new_h=Harness(cell,tmp_path/'new')
    monkeypatch.setattr(mechanics,'plugin',old)
    old_h=Harness(cell,tmp_path/'previous')
    results=[]
    for method in METHODS:
        suffix='' if method=='off' else '_ir32'
        path=R11/'artifacts'/f'r11_{cell}_{method}{suffix}.pkl'
        m=pickle.loads(path.read_bytes())['method']
        a=new_h.run(m,tasks=range(2),steps=12,force_guards=True)
        b=old_h.run(m,tasks=range(2),steps=12,force_guards=True)
        assert a[:2]==b[:2]
        results.append(dict(method=method,decisions=24,action_bytes_identical=True,decision_bytes_identical=True))
    EVIDENCE.setdefault('previous_plugin_diff',[]).append(dict(model=model,methods=results))
    new_h.rt.flush_all();old_h.rt.flush_all()


def test_existing_recordings_default_off_unchanged_bytes(tmp_path):
    from exp.offline_search.closed_loop.test_look_recording import writer_options,connection
    old=previous('look_recording')
    rows=[]
    for i,module in enumerate((old,look_recording)):
        root=tmp_path/str(i);root.mkdir()
        opts,_=writer_options(root,shadow=True,tokens=True,images=56)
        writer=module.LookWriter(opts,'pi05');conn,s,obs,_=connection(plugin,opts,writer)
        conn.infer({**obs,'_miss':True});record=list(module.read_looks(writer.directory))[0]
        rows.append(record)
    assert set(rows[0])==set(rows[1])
    # Root/UUID shard headers describe identity/storage. All content arrays,
    # guard state, distance semantics and shadow metadata retain exact bytes.
    for field in set(rows[0])-{'meta','guard_state'}:
        a,b=rows[0][field],rows[1][field]
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    assert rows[0]['guard_state']==rows[1]['guard_state']
    for field in set(rows[0]['meta'])-{'t_start'}:
        assert rows[0]['meta'][field]==rows[1]['meta'][field]
    EVIDENCE['previous_recording_diff']=dict(default_off=True,array_bytes_identical=True,fields=len(rows[0]))
