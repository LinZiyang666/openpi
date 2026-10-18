"""Freeze new legacy orchestration beside existing islands; never replace them."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile

from .bundle import files
from .node import atomic, read
from .switch import enabled

REPO = Path(__file__).resolve().parents[5]
EXTEND = b'from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n'


def build(root, *, checkout=REPO):
    checkout, root = Path(checkout), Path(root)
    names = files() + ['exp/offline_search/rounds/r17/autoscale_1017/legacy_release.py']
    names += [str(p.relative_to(checkout)) for p in sorted((checkout/'src/openpi/conductor').glob('*.py'))]
    names += ['exp/offline_search/rounds/r16/'+n+'.py' for n in ('conductor','conductor_runner')]
    names += ['exp/r17_integration/ops/'+n+'.py' for n in ('formal_process','worker_gpu','worker_memory','timan1_workers','local_workers')]
    names += ['exp/offline_search/closed_loop/ops/h100/'+n for n in ('run_arm.sh','node.py')]
    content = {n:(checkout/n).read_bytes() for n in sorted(set(names))}
    # Namespace extension preserves frozen simulation/model/init packages.
    content.update({n:EXTEND for n in ('src/openpi/__init__.py','exp/__init__.py','exp/offline_search/__init__.py','exp/offline_search/rounds/r16/__init__.py','exp/offline_search/rounds/r17/filter_failed/__init__.py')})
    content['sitecustomize.py'] = ("import os,sys\nfrom pathlib import Path\nimport exp,exp.offline_search,openpi\nr=Path(__file__).parent\nexp.__path__.insert(0,str(r/'exp'))\nexp.offline_search.__path__.insert(0,str(r/'exp/offline_search'))\nopenpi.__path__.insert(0,str(r/'src/openpi'))\n").encode()
    hashes = {n:hashlib.sha256(b).hexdigest() for n,b in content.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    folder = root/'autoscale_releases'/digest
    for n, data in content.items():
        p = folder/n
        if p.exists():
            if p.read_bytes() != data:raise ValueError('immutable autoscale overlay changed: '+str(p))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    archive = folder.with_suffix('.tar')
    if not archive.exists():
        with tarfile.open(archive, 'w') as out:
            for n in sorted(content):out.add(folder/n, arcname=n, recursive=False)
    return dict(digest=digest, local=str(folder), archive=str(archive), files=hashes,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())


def stage(root, host, *, pipeline, maximum, base, hard_limits=None):
    """Called only by explicit future prepare/launch, never by imports/tests."""
    if not enabled():return None
    from exp.r17_integration.ops.formal_eval import transfer
    from exp.offline_search.closed_loop.ops.h100 import control
    from .defaults import desired_maxima
    maximum = desired_maxima('libero', {host: maximum}, hard_limits)[host]
    record = build(root)
    remote = '/scratch/zixuans8/openpi_autoscale_releases/'+record['digest']
    temporary = '/tmp/openpi_autoscale_'+record['digest']+'.tar'
    transfer(['rsync','push',record['archive'],host+':'+temporary])
    script = ('test "$(sha256sum '+shlex.quote(temporary)+" | cut -d ' ' -f 1)\" = "+record['archive_sha256']+
              ' && mkdir -p '+shlex.quote(remote)+' && tar --skip-old-files -xf '+shlex.quote(temporary)+' -C '+shlex.quote(remote))
    code = 'import hashlib;from pathlib import Path;p=Path('+repr(remote)+');pins='+repr(record['files'])+';assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in pins.items()), \"immutable autoscale overlay mismatch\"'
    script += ' && /usr/bin/python3 -B -S -c ' + shlex.quote(code)
    control.remote(host, ['bash','-c',script])
    record.update(remote=remote, host=host, pipeline=pipeline, maximum=maximum, base=base, mode='dynamic',
        node_config=json.loads(os.environ.get('OPENPI_AUTOSCALE_NODE_CONFIG', '{}')))
    atomic(Path(root)/'LEGACY_AUTOSCALE.json', record)
    return record


def rpc_command(helper, action, target, spec):
    """Fresh node helper + run_arm source, retaining native ownership markers."""
    overlay = spec.get('env', {}).get('OPENPI_AUTOSCALE_OVERLAY')
    if spec.get('role') != 'driver' or not overlay:
        return ['python3',helper,action,target]
    node = overlay+'/exp/offline_search/closed_loop/ops/h100/node.py'
    boot = ('import importlib.util;from pathlib import Path;'
            's=importlib.util.spec_from_file_location("autoscale_node",'+repr(node)+');'
            'n=importlib.util.module_from_spec(s);s.loader.exec_module(n);'
            'n.ISLAND=Path('+repr(str(Path(helper).parent.parent))+')\n')
    scope = os.environ.get('FFK_WORKER_FLEET_ROOT')
    if scope:
        import inspect
        from exp.offline_search.rounds.r17.filter_failed_knob.vla_shared import configure_node, worker_lock
        boot += inspect.getsource(configure_node)+'\nconfigure_node(n,'+repr(worker_lock(scope, Path(helper).parent.parent))+','+repr(Path(scope).name)+')\n'
    boot += 'n.main()\n'
    return ['/usr/bin/python3','-B','-S','-c',boot,action,target]


def coordinator(root, argv):
    """Re-exec the R16 wrapper once so imports select sealed new ops code.

    Its remaining scientific modules still resolve from the original server
    checkout. This function is only reachable from an explicit launch command.
    """
    if not enabled():return None
    record = read(Path(root)/'LEGACY_AUTOSCALE.json')
    if not record:raise ValueError('prepare legacy autoscale release first')
    if os.environ.get('OPENPI_AUTOSCALE_COORDINATOR') == record['digest']:return record
    env = dict(os.environ, OPENPI_AUTOSCALE_COORDINATOR=record['digest'],
               OPENPI_AUTOSCALE_REMOTE=record['remote'])
    local = record['local']
    env['PYTHONPATH'] = local+':'+local+'/src:'+env.get('PYTHONPATH','')+':'+str(REPO)
    boot = ('from exp.offline_search.rounds.r17.filter_failed_knob.vla_workers import run_r16_selected;run_r16_selected()')
    # -c keeps sys.path[0] == cwd. The overlay must also precede that frozen
    # package, hence put its paths first before importing the selected wrapper.
    boot = 'import sys;sys.path[:0]='+repr([local,local+'/src'])+';'+boot
    os.execve(sys.executable, [sys.executable,'-c',boot,*argv[1:]], env)
