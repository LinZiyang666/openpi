"""Print (default) or install (--execute, coordinator only) the serving modules round 6 needs in the h100 tree.

New files: explore_opus/round6/{__init__,methods}.py.  Already-deployed dependencies are listed with their local sha so
the coordinator can verify the h100 copies byte for byte (installation refuses to overwrite a differing file):
explore_opus/{__init__,round2/__init__,round2/methods}.py (EscalateOnlyNP) and fable's
explore_fable/{__init__,round2/__init__,round2/tools/__init__,round2/tools/methods,round3/__init__,round3/tools/__init__,
round3/tools/methods}.py (CorrectedCache, CorrectedCacheJ, the r3c stacks).
"""
import argparse
import hashlib
from pathlib import Path

REPO = Path("/home/weiland/projects/openpi")
RUN = Path("/home/weiland/trace_runs/os_closed_loop/r09_opus_r6")
R9 = Path("exp/offline_search/rounds/r09")
FILES = [R9 / "explore_opus/__init__.py", R9 / "explore_opus/round6/__init__.py", R9 / "explore_opus/round6/methods.py",
         R9 / "explore_opus/round2/__init__.py", R9 / "explore_opus/round2/methods.py",
         R9 / "explore_fable/__init__.py", R9 / "explore_fable/round2/__init__.py",
         R9 / "explore_fable/round2/tools/__init__.py", R9 / "explore_fable/round2/tools/methods.py",
         R9 / "explore_fable/round3/__init__.py", R9 / "explore_fable/round3/tools/__init__.py",
         R9 / "explore_fable/round3/tools/methods.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    for rel in FILES:
        print(sha(REPO / rel), rel, "->", Path("/data/oscl_h100/openpi") / rel)
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
        for rel in FILES:
            src = REPO / rel
            digest = sha(src)
            stage = c.BASE / "runs" / RUN.name / "source_install" / f"{digest}_{src.name}"
            c.remote("h100", ["mkdir", "-p", stage.parent])
            c.push("h100", src, stage)
            print(c.remote("h100", ["python3", "-c", program, stage, c.BASE / "openpi" / rel, digest]))


if __name__ == "__main__":
    main()
