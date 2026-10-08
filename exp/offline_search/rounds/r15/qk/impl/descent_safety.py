"""Default-off source proof and conservative descent server import inventory.

This inspects B-root definitions and Python source only. It neither touches a
running process nor asserts which module versions are already in its memory.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SHARED = {
    'exp/offline_search/closed_loop/plugin.py': 'server runtime: both serve_pi05 and serve_groot',
    'exp/offline_search/closed_loop/ops/h100/assets.py': 'shared dependency planner; also in conservative server import closure',
    'exp/offline_search/rounds/r04/k4_eval/cost_ledger.py': 'collector cost accounting; not policy server runtime',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DefaultOff(ast.NodeTransformer):
    """Specialize only the documented QK switch and QK-private state."""
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in ('os_light_key', '_qk_full')):
            return ast.Constant(False)
        if (isinstance(node.func, ast.Name) and node.func.id == 'any'
                and "'qk_work'" in ast.unparse(node)):
            return ast.Constant(False)
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and node.left.value == '--os-light-key':
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.NotIn):
                return ast.Constant(True)
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
                return ast.Constant(False)
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Constant):
            return ast.Constant(not node.operand.value)
        return node

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        identity = isinstance(node.op, ast.And)
        values = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                if bool(value.value) != identity:
                    return ast.Constant(not identity)
            else:
                values.append(value)
        if not values:
            return ast.Constant(identity)
        if len(values) == 1:
            return values[0]
        node.values = values
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_Expr(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'add_argument' and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == '--os-light-key'):
            return None  # Added opt-in parser option, absent in descent invocations.
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'audit_view' and isinstance(node.value, ast.Name)
                and node.value.id == 'view'):
            return None  # Default branch introduces an identity local alias.
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == '_try_blind':
            aliases = [n for n in node.body if isinstance(n, ast.Assign)
                       and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id == 'kd']
            if aliases:
                assert len(aliases) == 1
                assignment = aliases[0]
                assert ast.unparse(assignment.value) == 's.rt.dims.KEY_DIM'
                class KeyDimension(ast.NodeTransformer):
                    def visit_Name(self, value):
                        return copy.deepcopy(assignment.value) if value.id == 'kd' else value
                node.body.remove(assignment)
                node = KeyDimension().visit(node)
        return node

    def visit_Name(self, node):
        if node.id == 'audit_view':
            node.id = 'view'
        return node


def normalized(path):
    tree = DefaultOff().visit(ast.parse(Path(path).read_text()))
    return ast.fix_missing_locations(tree)


def resolve(module):
    if module.startswith('openpi_client'):
        root = REPO/'packages/openpi-client/src'
    elif module == 'openpi' or module.startswith('openpi.'):
        root = REPO/'src'
    else:
        root = REPO
    path = root.joinpath(*module.split('.'))
    for candidate in (path.with_suffix('.py'), path/'__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def closure(seeds):
    pending, seen = list(seeds), set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        rel = path.relative_to(REPO)
        if rel.parts[0] == 'src':
            rel = rel.relative_to('src')
        elif rel.parts[:3] == ('packages','openpi-client','src'):
            rel = rel.relative_to('packages/openpi-client/src')
        package = list(rel.parts[:-1])
        for parent in path.parents:
            if parent == REPO:
                break
            init = parent/'__init__.py'
            if init.is_file() and init not in seen:
                pending.append(init)
        for node in ast.walk(normalized(path)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    prefix = package[:len(package)-node.level+1]
                    base = '.'.join(prefix+([base] if base else []))
                modules = [base]+[base+'.'+alias.name for alias in node.names if alias.name != '*']
            for module in modules:
                target = resolve(module)
                if target is not None and target not in seen:
                    pending.append(target)
    assert not any('/r15/qk/impl/' in str(path) for path in seen)
    return sorted(seen)


def run():
    shared = []
    for name, role in SHARED.items():
        before = HERE/'before'/Path(name).name
        after = REPO/name
        old = ast.dump(normalized(before), include_attributes=False)
        new = ast.dump(normalized(after), include_attributes=False)
        if old != new:
            # Save reviewable normalized sources, never mutate shared files.
            (HERE/f'default_off_before_{Path(name).name}.txt').write_text(ast.unparse(normalized(before)))
            (HERE/f'default_off_after_{Path(name).name}.txt').write_text(ast.unparse(normalized(after)))
            raise AssertionError('default-off AST differs: '+name)
        shared.append(dict(path=name,role=role,before_sha256=sha(before),sha256=sha(after),
                           default_off_AST_equal=True,normalized_AST_sha256=hashlib.sha256(new.encode()).hexdigest()))
    directory = REPO/'exp/offline_search/rounds/r15/descent'
    roots = json.loads((directory/'dev_roots.json').read_text())+json.loads((directory/'c10_roots.json').read_text())
    seeds = [REPO/'exp/offline_search/closed_loop'/name
             for name in ('plugin.py','serve_pi05.py','serve_groot.py','stage_overrides.py')]
    seeds.append(REPO/'scripts/serve_policy.py')  # Dynamically loaded by serve_pi05.
    arms, fits = [], {}
    for record in roots:
        assert record['init_pool'] == 'B'
        path = Path(record['root'])/'arms.json'
        for row in json.loads(path.read_text()):
            assert '--os-light-key' not in row['plugin_args']
            assert not any(arg.startswith('--os-light-key=') for arg in row['plugin_args'])
            arms.append(dict(root=record['root'],arm=row['arm'],model=row['model'],
                             definition_sha256=sha(path),os_light_key=False))
            module = row['method'].split(':')[0]
            target = resolve(module)
            if target:
                seeds.append(target)
        for path, digest in record['fit_sha256'].items():
            assert sha(path) == digest, path
            fits[path] = digest
    code = closure(seeds)
    previous = json.loads((HERE/'before_final/deployment_manifest.json').read_text())
    previous_code = {f['path']:f['sha256'] for f in previous['code_files']}
    serving_names = list(SHARED)+['exp/offline_search/rounds/r15/qk/impl/'+name for name in
                         ('__init__.py','encoder.py','recipe.py','runtime.py','recording.py','accounting.py')]
    serving_unchanged = {name:sha(REPO/name)==previous_code[name] for name in serving_names}
    assert all(serving_unchanged.values())  # Final selection/refit changed no serving code.
    files = [dict(path=str(p.relative_to(REPO)),sha256=sha(p),
                  unchanged_since_round1=(previous_code[str(p.relative_to(REPO))]==sha(p)
                      if str(p.relative_to(REPO)) in previous_code else None)) for p in code]
    server_paths = {f['path'] for f in files}
    for record in shared:
        record['direct_policy_server_import'] = record['path'].endswith('/closed_loop/plugin.py')
        record['conservative_server_dependency'] = record['path'] in server_paths
    report = dict(PASS=True,scope='configured B descent arms and conservative server import closure; no live process inspection',
        assumptions=['--os-light-key absent/false; QK-private _qk_full never set on this path',
                     'descent decision rows do not contain qk_work',
                     'AST specialization removes the added CLI option and resolves audit_view=view and blind kd=KEY_DIM aliases'],
        shared_changed_files=shared,shared_server_files=files,configured_arms=arms,descent_fit_hashes=fits,
        changed_shared_server_files=[f for f in shared if f['conservative_server_dependency']],
        shared_server_changed_since_round1=[f for f in files if f['unchanged_since_round1'] is False],
        shared_server_unrecorded_in_prior_inventory=[f for f in files if f['unchanged_since_round1'] is None],
        serving_files_unchanged_during_final_swap=serving_unchanged,
        QK_runtime_imports_when_disabled=0,
        deployment_notes=['Current interpreter modules retain loaded code; copying a file does not reload it.',
            'Future descent server starts load the shared plugin with default-off equivalent logic.',
            'Use complete atomic file replacement, without reloads/restarts or edits to descent roots/fits.',
            'Student/refit/root paths are separate from existing descent assets; do not overwrite their fits.',
            'Plugin is directly imported by both server entry points. Assets is shared planner tooling and a potential '
                'transitive server dependency; cost_ledger is collector tooling, absent from this server closure.'],
        servers=0,GPU=0,remote_operations=0,test_A_reads=0)
    (HERE/'descent_safety.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DESCENT_DEFAULT_OFF_PASS',len(arms),'configured arms;',len(files),'shared source files')
    return report


if __name__=='__main__':
    run()
