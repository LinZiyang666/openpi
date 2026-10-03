"""Print (default) or install (--execute, coordinator only) the serving modules R9Recipe needs in the h100 tree.

Serving needs only ``rounds/r09/recipe/{__init__,recipe}.py`` plus non-exploratory round modules that are already on
h100 (listed under VERIFY with their local sha: R4 BlindAWM / wrist / WristView, R6 FitUnpickler, R7 StageWrist /
StageFollow / StageTable, R8 judges).  ``--execute`` installs FILES only and refuses to overwrite a differing file.
"""
import argparse
import hashlib
from pathlib import Path

REPO = Path("/home/weiland/projects/openpi")
RUN = Path("/home/weiland/trace_runs/os_closed_loop/r09_recipe_eq")
R9 = Path("exp/offline_search/rounds/r09")
FILES = [R9 / "recipe/__init__.py", R9 / "recipe/recipe.py"]
RR = Path("exp/offline_search/rounds")
VERIFY = [RR / "r04/k1_blind/blind_awm.py", RR / "r04/k1_blind/wrist.py", RR / "r04/k3_cost/method.py",
          RR / "r06/ideation_Q1/method_c/common.py", RR / "r07/c2_wrist/method.py", RR / "r07/c1_follow/methods.py",
          RR / "r07/stages/stages.py", RR / "r08/abl/judge.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    for rel in FILES:
        print(sha(REPO / rel), rel, "->", Path("/data/oscl_h100/openpi") / rel)
    for rel in VERIFY:
        print(sha(REPO / rel), rel, "(verify only; already deployed)")
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
