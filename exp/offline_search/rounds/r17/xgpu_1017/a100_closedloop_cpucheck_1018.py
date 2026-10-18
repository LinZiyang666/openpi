"""CPU replay of the actual frozen formal planners against all sealed UID subsets."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from a100_closedloop_prepare_1018 import HERE,OUT,sha,write


def check(model,arm):
 base=OUT/model/arm;pr=json.loads((base/'PREREGISTRATION.json').read_text())
 c=json.loads((base/'campaign_template.json').read_text())
 c.update(runtime=pr['runtime_source'],local_assets=str(base/'assets'),assets=str(base/'assets'),xgpu_subset=str(base/'PLAN.json'))
 import a100_closedloop_run_1018 as run
 run.paths(c)
 if sha(base/'PLAN.json')!=pr['plan_sha256']:raise ValueError('preregistered plan changed')
 bm,spec,tasks,endpoints=run.filtered_plan(c)(c)
 expected={t['task_uid']:t for t in json.loads((base/'PLAN.json').read_text())['tasks']}
 if len(tasks)!=pr['episodes'] or {t.task_uid for t in tasks}!=set(expected):raise ValueError('population changed')
 for task in tasks:
  actual=asdict(task)
  for key in actual:
   if key not in ('server_host','server_port') and actual[key]!=expected[task.task_uid][key]:raise ValueError('source scientific fields changed')
 ref=json.loads((base/'REFERENCE.json').read_text())['outcomes']
 if set(ref)!=set(expected) or any(r.get('accepted') is not True or r.get('error') for r in ref.values()):
  raise ValueError('accepted formal reference incomplete')
 return dict(model=model,arm=arm,episodes=len(tasks),original_uid_equal=True,scientific_fields_equal=True,
             accepted_reference_complete=True,plan_sha256=pr['plan_sha256'],runtime=pr['runtime_identity'],gpu_initialized=False)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--arm');a=p.parse_args()
 if a.model:print(json.dumps(check(a.model,a.arm)))
 else:
  reports=[]
  for model in ('pi05','groot','dp','fastwam','cosmos'):
   for arm in ('pure','cache'):
    result=subprocess.run([sys.executable,__file__,'--model',model,'--arm',arm],text=True,capture_output=True,check=True)
    reports.append(json.loads(result.stdout))
  write(OUT/'CPU_CHECKS.json',reports);print(json.dumps(reports,indent=2))
