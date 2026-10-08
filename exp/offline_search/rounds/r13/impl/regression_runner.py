"""Run existing suites while redirecting their scratch writes to R13 impl.

R12 unittest modules contain literal HERE/scratch paths. A symlinked read view
keeps every original artifact/config read intact and puts only scratch under our
allowed output tree. No old round file or active root is modified.
"""
from pathlib import Path
import json
import shutil
import pytest

from exp.offline_search.rounds.r12.lookhalf import selftest as lookhalf
from exp.offline_search.rounds.r12.guard import selftest as guard
from .prepare import HERE, replace_owned


def view(module, name):
    real=module.HERE
    output=HERE/'regression_view'/name
    output.mkdir(parents=True,exist_ok=True)
    for p in real.iterdir():
        dest=output/p.name
        if p.name=='scratch':
            dest.mkdir(exist_ok=True)
        elif p.name=='coordinator_commands.sh':
            if dest.is_symlink():
                dest.unlink()
            dest.write_text(p.read_text().replace(str(real/'pilots'),str(output/'pilots')))
        elif not dest.exists():
            dest.symlink_to(p,target_is_directory=p.is_dir())
    module.HERE=output
    for sibling in ('astra','opus'):
        target=output.parent/sibling
        if not target.exists():target.symlink_to(real.parent/sibling,target_is_directory=True)


class IsolateScratch:
    def pytest_collection_modifyitems(self, session, config, items):
        """Patch the modules pytest actually loaded, including namespace aliases."""
        seen=set()
        for item in items:
            module=item.module
            if id(module) in seen:continue
            seen.add(id(module));source=Path(module.__file__).resolve()
            if source.name=='selftest.py' and source.parent.name in ('lookhalf','guard'):
                view(module,source.parent.name)
            if source.name=='test_cache_ablate.py':
                # This old suite assumes pilot roots have never run. Live roots
                # already contain runs/state; copy only their definitions and
                # execute the unchanged assertions against that isolated copy.
                for record in module.RECORDS:
                    old=Path(record['root']);root=HERE/'regression_roots'/old.name
                    root.mkdir(parents=True,exist_ok=True)
                    for name in ('arms.json','arms_in.json','ROOT.json','dev_selection.json','eval500.json'):
                        p=old/name
                        if p.is_file():
                            if name in ('dev_selection.json','eval500.json'):
                                shutil.copyfile(p,root/name)
                            else:
                                (root/name).write_text(json.dumps(replace_owned(json.loads(p.read_text()),old,root),indent=1)+'\n')
                    shutil.copytree(old/'config',root/'config',dirs_exist_ok=True)
                    changed=replace_owned(record,old,root)
                    record.clear();record.update(changed)


if __name__=='__main__':
    view(lookhalf,'lookhalf');view(guard,'guard')
    files=[str(p) for p in sorted((HERE.parents[2]/'closed_loop').glob('test_*.py'))]
    files += ['exp/offline_search/rounds/r11/knob/tests.py',
              'exp/offline_search/rounds/r12/lookhalf/selftest.py',
              'exp/offline_search/rounds/r12/guard/selftest.py']
    raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',*files,'--basetemp='+str(HERE/'pytest_regression'),
                                 '--junitxml='+str(HERE/'regression_tests.xml')],plugins=[IsolateScratch()]))
