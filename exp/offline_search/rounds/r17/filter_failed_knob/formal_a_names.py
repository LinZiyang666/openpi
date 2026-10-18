"""Short formal-A launch names and their single authoritative alias map."""
from pathlib import Path
from .common import HERE, EVAL, read

MAP = HERE/'formal_a_1016/NAME_MAP.json'
MODELS = dict(dp='dp',fastwam='fw',cosmos='co',pi05='pi',groot='gr')
MODES = dict(full='f',reduced='r')
ARMS = dict(original='u',filtered='f')
STAGES = dict(off='o',off_complete='c',levels='l')


def unit(model,mode,arm,stage):
    return f'fa_{MODELS[model]}_{MODES[mode]}_{ARMS[arm]}_{STAGES[stage]}'


def cell_name(unit_name,cell):
    _,suite,size=cell.split('_')
    return unit_name+'_'+('l' if suite=='l10' else 's')+size


def resolve_name(name):
    return read(MAP)['names'].get(name,name) if MAP.exists() else name


def resolve_reference(root,arm=None):
    if not MAP.exists():return root,arm
    data=read(MAP)
    return data['runs'].get(str(root),str(root)),data['names'].get(arm,arm)


def resolve_job(job):
    if job.get('reuse_only'):return job
    root,_=resolve_reference(job['run'])
    return dict(job,name=resolve_name(job['name']),run=str(root))
