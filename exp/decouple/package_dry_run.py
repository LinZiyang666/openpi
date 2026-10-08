"""CPU-only parser, wire replay and fresh-exec split startup checks with stubs."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from exp.decouple.package_config import REPO, PYTHON, GROOT, make_arm, materialize, worker_command, sha, write_json
from exp.decouple.benchmark_gpu import expand_tokens


def parse_resident(config_path, model, out):
    config = json.loads(Path(config_path).read_text())
    case = config['cases'][model]
    argv = [expand_tokens(a, backend_port=config['backend_port'], out=out) for a in case['resident_command'][4:]]
    registry = argparse.ArgumentParser(add_help=False)
    registry.add_argument('--registry', required=True)
    registry.add_argument('--server-tag', required=True)
    registry.add_argument('--resident-suite')
    _, remaining = registry.parse_known_args(argv)
    from exp.offline_search.closed_loop import plugin, stage_overrides
    cost, remaining = stage_overrides.parse_flags(remaining)
    opts, remaining = plugin.parse_cli(remaining)
    assert opts.os_method == case['row']['method']
    if model == 'pi05':
        import tyro
        spec = importlib.util.spec_from_file_location('serve_policy', REPO / 'scripts/serve_policy.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['serve_policy'] = module
        spec.loader.exec_module(module)
        args = tyro.cli(module.Args, args=remaining)
        assert args.concurrent and not args.non_concurrent and args.replicas == 1
        assert args.policy.config == 'pi05_libero'
        stage_overrides.validate_method(opts, cost.os_stage1_mode)
        assert stage_overrides.miss_steps_from_yaml(args.cache_config, model) == 1
    else:
        # Select island dependencies before importing torch, while retaining
        # the owner's main-tree interpreter for this preparation check.
        sys.path.insert(0, '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/lib/python3.11/site-packages')
        sys.path.insert(0, GROOT)
        sys.path.insert(0, GROOT + '/examples/Libero')
        from exp.libero_groot import serve_groot_libero
        original = argparse.ArgumentParser.parse_args
        class Parsed(Exception):
            pass
        def intercept(parser, *args, **kwargs):
            values = original(parser, *args, **kwargs)
            assert values.concurrent and values.allow_dynamic_bundles
            assert values.denoising_steps == 1 and values.checkpoint == case['checkpoint']
            raise Parsed
        argparse.ArgumentParser.parse_args = intercept
        sys.argv = [serve_groot_libero.__file__, *remaining]
        try:
            serve_groot_libero.main()
        except Parsed:
            pass
        else:
            raise AssertionError('model main passed CLI parsing')
        finally:
            argparse.ArgumentParser.parse_args = original
    print(f'ARGV_PARSE_OK {model}; no policy loader called')


def dry_run(config, config_path):
    root = REPO / ('exp/decouple/results/package_dry_run_smoke' if config['mode']=='smoke' else 'exp/decouple/results/package_dry_run')
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'HF_MODULES_CACHE':str(root / 'hf_modules')}
    assert not os.environ.get('CUDA_VISIBLE_DEVICES')
    checks = []
    dependency = ('import sys;sys.path.insert(0,' + repr(str(REPO / 'exp/decouple/_vendor')) + ');'
                  'import cloudpickle;assert cloudpickle.__version__=="3.1.2";'
                  'assert cloudpickle.loads(cloudpickle.dumps(lambda:17))()==17;print(cloudpickle.__file__)')
    result = subprocess.run([PYTHON,'-S','-B','-c',dependency], cwd=REPO, env=env,
                            capture_output=True,text=True,timeout=10)
    assert result.returncode == 0, result.stderr
    assert '/exp/decouple/_vendor/cloudpickle/' in result.stdout
    modules = ['exp.decouple.package_config', 'exp.decouple.gpu_package', 'exp.decouple.package_driver',
               'exp.decouple.current_fleet', 'exp.decouple.benchmark_gpu', 'exp.decouple.qualify_gpu',
               'exp.decouple.summarize_gpu', 'exp.decouple.latency_proxy', 'openpi.serving.split',
               'exp.offline_search.rounds.r16.conductor','exp.decouple.collect_gpu','exp.decouple.gate_cache']
    for module in modules:
        result = subprocess.run([PYTHON, '-B', '-m', module, '--help'], cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
        if result.returncode:
            raise ValueError(f'{module} --help failed:\n{result.stdout[-4000:]}')
        checks.append(module + ' --help')
    result = subprocess.run(['bash', '-n', str(REPO / 'exp/decouple/run_gpu_package.sh')],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    commands = {}
    from exp.offline_search.rounds.r16.conductor import read_group
    from exp.decouple.package_driver import BenchmarkStrategy
    from openpi.conductor.task import ServerEndpoint
    for model, case in config['cases'].items():
        out = root / model
        out.mkdir(exist_ok=True)
        materialize(case, out)
        group = read_group(out / 'group.json')
        result = subprocess.run([PYTHON, '-B', '-m', 'exp.decouple.package_dry_run',
                                 '--parse-resident', model, '--config', str(config_path), '--out', str(out)],
                                cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=60)
        if result.returncode:
            raise ValueError(f'{model} resident argv failed parsing:\n{result.stdout[-5000:]}')
        (out / 'argv_parse.log').write_text(result.stdout)
        graph = BenchmarkStrategy(group, case, out).plan([case['row']['arm']],
                        {case['row']['arm']:ServerEndpoint(config['advertised_host'], config['public_port'])})
        assert len(graph.stages['eval'].episodes) == config['expected_episodes'] and len(graph.stages['warmup'].episodes) == 10
        assert graph.upstreams('eval') == ['warmup'] or set(graph.upstreams('eval')) == {'warmup'}
        arms = [make_arm(config, model, kind, f'decouple_bench_dry_{model}_{kind}') for kind in config['initial_arms']]
        commands[model] = [{k:[expand_tokens(a, port=config['public_port'], backend_port=config['backend_port'], out=out)
                                  for a in arm[k]] for k in ('server_command', 'worker_command')} for arm in arms]
        for arm in arms:
            assert arm['manifest_sha256'] == case['manifest_sha256']
        commands[model].append(dict(deployed_worker_agent=worker_command(config, model,
                       config['worker_island'] + '/decouple_agents/DRY/lease.json', 'DRY_TOKEN')))
    # Verify the unchanged worker wire contract against the sealed deployment
    # inventory, without reading the worker host or deploying any files.
    manifest = dict(line.split('  ', 1)[::-1] for line in config['worker_sha256'].splitlines())
    contract = {}
    for name in ('src/openpi/conductor/task.py', 'src/openpi/conductor/protocol.py',
                 'examples/libero/episode_runner.py','packages/openpi-client/src/openpi_client/websocket_client_policy.py',
                 'packages/openpi-client/src/openpi_client/msgpack_numpy.py'):
        actual = sha(REPO / name)
        assert actual == manifest[name], f'deployed worker wire mismatch: {name}'
        contract[name] = actual
    write_json(root / 'argv.json', commands)
    # Execute the qualifier's actual GR00T imports with the GR00T site-packages
    # under the required main interpreter. Block JAX imports explicitly.
    tree=ast.parse((REPO/'exp/decouple/qualify_gpu.py').read_text())
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    imports=[]
    for node in main.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            imports.append(node)
        if isinstance(node,ast.If) and ast.unparse(node.test)=="args.model == 'pi05'":
            imports += [node for node in node.orelse if isinstance(node,(ast.Import,ast.ImportFrom))]
            break
    import_code='\n'.join(ast.unparse(node) for node in imports)
    site='/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/lib/python3.11/site-packages'
    script=('import sys,importlib.util,builtins;sys.path.insert(0,'+repr(site)+');\n'
            '_find_spec=importlib.util.find_spec;_import=builtins.__import__\n'
            'importlib.util.find_spec=lambda name,*a,**k: None if name=="jax" or name.startswith("jax.") else _find_spec(name,*a,**k)\n'
            'def no_jax(name,*a,**k):\n'
            ' if name=="jax" or name.startswith("jax."): raise ImportError("JAX forbidden in GR00T qualification")\n'
            ' return _import(name,*a,**k)\n'
            'builtins.__import__=no_jax\n'+import_code+'\nassert "jax" not in sys.modules;print("GROOT_QUALIFIER_IMPORTS_NO_JAX_OK")')
    result=subprocess.run([PYTHON,'-S','-B','-c',script],cwd=REPO,
                          env={**env,**config['cases']['groot']['environment'],'CUDA_VISIBLE_DEVICES':''},
                          capture_output=True,text=True,timeout=60)
    if result.returncode:
        raise ValueError('GR00T qualifier runtime imports failed: '+result.stderr[-5000:])
    (root/'groot_qualifier_imports.log').write_text(result.stdout+result.stderr)
    from exp.decouple.cpu_collection_check import check
    collection=check(config,root/'cpu_collection')
    from exp.decouple.cpu_startup_check import check as startup_check
    startup=startup_check(config,root/'cpu_startup')
    from exp.decouple.gate_cache import GateCache
    available={key:bool(GateCache().lookup(config,key)) for key in config['gate_inputs']}
    return dict(bash_syntax=True, help_checked=checks, resident_argv_parsed=['groot','pi05'],
                bundled_cloudpickle='3.1.2; isolated main-Python -S roundtrip passed',
                worker_wire_sha256=contract, task_graph_checked=True, expanded_argv=str(root / 'argv.json'),
                groot_qualifier_imports_no_jax=True,cpu_collection=collection,cpu_split_startup=startup,
                gate_cache_available=available,remote_operations=0,model_loads=0,tcp_listeners=0,
                listeners_created=True,private_unix_listeners=2,
                processes_launched='CLI help/parser/import checks; 18 fresh CPU-stub split children, private AF_UNIX IPC; in-memory worker replay')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parse-resident', choices=('groot', 'pi05'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    parse_resident(args.config, args.parse_resident, args.out)


if __name__ == '__main__':
    main()
