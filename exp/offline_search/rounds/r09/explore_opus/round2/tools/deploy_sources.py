"""Print (default) or install (--execute, coordinator only) the serving module into the isolated h100 tree.

Installs only new files under exp/offline_search/rounds/r09/explore_opus/ ; refuses to overwrite a differing
existing remote file (hard link + SHA check).  Mirrors astra's round-2 deploy tool; nothing else is touched.
"""
import argparse
import hashlib
from pathlib import Path

from .common import HERE, RUNS

RUN = RUNS / "r09_opus_escalation"
REL = Path("exp/offline_search/rounds/r09/explore_opus")
FILES = [(HERE.parent / "__init__.py", REL / "__init__.py"),
         (HERE / "__init__.py", REL / "round2" / "__init__.py"),
         (HERE / "methods.py", REL / "round2" / "methods.py")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    for src, rel in FILES:
        print(sha(src), src, "->", Path("/data/oscl_h100/openpi") / rel)
    if not a.execute:
        return
    from exp.offline_search.closed_loop.ops.h100 import control as c
    program = '''import hashlib,os,sys
from pathlib import Path
source,target,digest=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
target.parent.mkdir(parents=True,exist_ok=True)
try: os.link(source,target)
except FileExistsError: pass
assert hashlib.sha256(target.read_bytes()).hexdigest()==digest, 'Refuse differing existing source'
print('SOURCE_READY',target,digest)
'''
    with c.fleet_lock(RUN):
        for src, rel in FILES:
            digest = sha(src)
            stage = c.BASE / "runs" / RUN.name / "source_install" / f"{digest}_{src.name}"
            target = c.BASE / "openpi" / rel
            c.remote("h100", ["mkdir", "-p", stage.parent])
            c.push("h100", src, stage)
            print(c.remote("h100", ["python3", "-c", program, stage, target, digest]))


if __name__ == "__main__":
    main()
