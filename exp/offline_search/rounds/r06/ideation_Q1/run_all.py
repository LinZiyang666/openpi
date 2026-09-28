import concurrent.futures,subprocess,pathlib,os,sys
from score import joblist
O=pathlib.Path(__file__).parent
CPUS='10-13,54-57'
def run(i):
 with open('/tmp/r6_Q1/job'+str(i)+'.log','w') as f:
  r=subprocess.run(['taskset','-c',CPUS,'.venv/bin/python',str(O/'score.py'),'--job',str(i)],stdout=f,stderr=subprocess.STDOUT)
 return i,r.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for i,code in pool.map(run,[i for i in range(len(joblist())) if i not in (1,2)]+[1,2]):print(i,code,flush=True)
