from pathlib import Path
B=Path(__file__).resolve().parent
Q=B.parent/'q2_groot'
R=B/'regression';(R/'dev').mkdir(parents=True,exist_ok=True);(R/'results/installed').mkdir(parents=True,exist_ok=True)
for p in (Q/'regression').rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.sh','.json') or 'results' in p.parts:continue
    if p.name=='install.py':continue
    s=p.read_text().replace(str(Q/'regression'),str(R)).replace('30-33,74-77','34-37,78-81').replace('/tmp/q2_','/tmp/q5_')
    # Keep source helpers in q5/dev and q5/before; installed label selects final candidate via env.
    if p.name=='launch_test.py':
        s=s.replace('def load(source):','def load(source):\n    source = os.environ.get("Q5_SOURCE", "dev") if source == "installed" else source')
        s=s.replace('    return plugin','    plugin._git_head = lambda: "q5-fixed-git-head"\n    return plugin')
    target=R/p.relative_to(Q/'regression');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(s)
# Direct-import drivers use a tiny owned boot import; no git invocation.
boot=B/'boot';boot.mkdir(exist_ok=True)
(boot/'sitecustomize.py').write_text('''import importlib.util, os, pathlib, sys
B=pathlib.Path(__file__).resolve().parent.parent
source=os.environ.get('Q5_SOURCE','dev')
for short in (('blind','plugin') if os.environ.get('Q5_GPU') else ('blind','plugin','verify_logs','selftest')):
    name='exp.offline_search.closed_loop.'+short
    p=B/source/(short+'.py') if source!='installed' else B.parents[2]/'closed_loop'/(short+'.py')
    if source=='installed':
        from exp.offline_search.closed_loop import plugin
        plugin._git_head=lambda:'q5-fixed-git-head'
        break
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    if short=='plugin':
        mod.REPO=next(p for p in B.parents if (p/'exp/trace_dual/config').is_dir())
        mod._git_head=lambda:'q5-fixed-git-head'
''')
# Add owned boot path to every subprocess environment used by copied tests.
for p in R.rglob('*'):
    if p.is_file() and p.suffix in ('.py','.sh'):
        p.write_text(p.read_text().replace('PYTHONPATH=.:src',f'PYTHONPATH={boot}:.:src'))
# Q2's new matrices exercise installed Q2 features through the final Q5 candidate.
T=B/'q2checks';(T/'results').mkdir(parents=True,exist_ok=True)
for name in ('new_tests.py','tail_parity.py','concurrency_test.py','contract_tests.py','cpu_transforms.py'):
    s=(Q/name).read_text().replace('30-33,74-77','34-37,78-81').replace('/tmp/q2_','/tmp/q5_q2_')
    s=s.replace("B/'regression/launch_test.py'",f"Path({str(R/'launch_test.py')!r})")
    s=s.replace("str(Path(__file__).resolve().parent/'regression')",repr(str(R)))
    s=s.replace("str(BASE/'regression')",repr(str(R)))
    s=s.replace('PYTHONPATH=.:src',f'PYTHONPATH={boot}:.:src')
    (T/name).write_text(s)
print(R)
