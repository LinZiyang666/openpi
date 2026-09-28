"""Frozen A-path reach for an explicitly exploratory pre-event probe, no SR forecast."""
from analyze import *

def main():
 results={}
 for cell in PAIRS:
  a=libload(cell);cal=json.loads((OUT/f'calibration_{cell}.json').read_text());A=read_arm(*PAIRS[cell][0]);p=json.loads((OUT/f'paired_{cell}.json').read_text());pb={(r['task'],r['init']):r for r in p['records']}
  rows=[]
  for key,ep in A['episodes'].items():
   first=None;first_features=None
   for d in ep['ds']:
    if not d.get('vision',True) or d['step']==0:continue
    # Uses only top-neighbor actions and future *library* rows, never future rollout actions.
    f=enrich(d,a,cal)
    if f['event_soon'] and f['disagreement']>1:
     first=d['step'];first_features=f;break
   b=pb[key];before=first is not None and (b['first_miss'] is None or first<b['first_miss'])
   rows.append(dict(task=key[0],init=key[1],first=first,features=first_features,before_B_first_miss=before,A_success=ep['success'],B_success=b['B_success'],B_first_miss=b['first_miss']))
  results[cell]=dict(episodes=len(rows),flagged=sum(r['first'] is not None for r in rows),flagged_A_success=sum(r['first'] is not None and r['A_success'] for r in rows),before_B_first_miss=sum(r['before_B_first_miss'] for r in rows),B_only_before=sum(r['before_B_first_miss'] and r['B_success'] and not r['A_success'] for r in rows),rows=rows)
  print(cell,{k:v for k,v in results[cell].items() if k!='rows'},flush=True)
 dump('event_probe.json',results)

if __name__=='__main__':main()
