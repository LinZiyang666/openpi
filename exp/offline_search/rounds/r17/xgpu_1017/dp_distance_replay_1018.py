"""CPU float32/float64 and cancellation-free replay of native first-head captures."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def weights(d2,z,meta):
 pos=z['pos'];order=pos[np.lexsort((pos,np.sqrt(np.maximum(d2[pos],0))))[:meta['k']]]
 d=np.sqrt(np.maximum(d2[order],0)).astype(np.float64);rel=d-d[0]
 w=np.exp(-(rel/max(rel[min(meta['kref'],len(rel))-1],1e-6))**2);w=(w/w.sum()).astype(np.float32)
 # Compare aligned donor IDs; sorting drift cannot disappear through rank-only comparison.
 return {int(z['rows'][i]):float(v) for i,v in zip(order,w)}


def replay(directory):
 rows=[]
 for p in sorted(Path(directory).rglob('*.json')):
  m=json.loads(p.read_text());npz=p.with_suffix('.npz')
  if 'npz_sha256' not in m:continue
  with npz.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['npz_sha256']
  with np.load(npz,allow_pickle=False) as f:z={k:f[k] for k in f.files}
  Z,A,n,y=[z[k] for k in ('Z','A0','n20','y')]
  native32=n-2.0*(Z@(A@y))+float(y@y)
  zz,aa,nn,yy=[v.astype(np.float64) for v in (Z,A,n,y)]
  float64=nn-2.0*(zz@(aa@yy))+float(yy@yy)
  u=zz@aa
  # n20 was fitted with the unquantized A, then stored as float32. Retain its
  # row residual so this formula is algebraically equivalent to the artifact.
  residual=nn-np.sum(u*u,axis=1)
  stable=np.sum((u-yy[None])**2,axis=1)+residual
  plain_euclidean=np.sum((u-yy[None])**2,axis=1)
  req=m['request'];key=tuple(req.get(k) for k in ('task_id','task_name','original_init_idx','seed','decision_seq','explicit_seed'))
  rows.append(dict(key=key,file=str(p),
      native32_vs_recorded_d2=float(np.max(abs(native32-z['d2']))),
      native32_vs_float64_d2=float(np.max(abs(native32-float64))),
      float64_vs_stable_d2=float(np.max(abs(float64-stable))),
      stored_norm_residual_max=float(np.max(abs(residual))),
      weights={name:weights(d,z,m) for name,d in [('recorded',z['d2']),('replay32',native32),
          ('float64_fixed_y',float64),('stable_fixed_y',stable),('euclidean_changes_artifact',plain_euclidean)]}))
 return rows


def main():
 p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('right');p.add_argument('--output',required=True);a=p.parse_args()
 left,right=replay(a.left),replay(a.right);by={}
 for side,rows in [('left',left),('right',right)]:
  for r in rows:by.setdefault(r['key'],{'left':[],'right':[]})[side].append(r)
 values=[]
 for key,pairs in by.items():
  if not pairs['left'] or not pairs['right']:continue
  metrics={}
  for name in pairs['left'][0]['weights']:
   errors=[]
   for x,y in itertools.product(pairs['left'],pairs['right']):
    u,v=x['weights'][name],y['weights'][name]
    errors.append(max(abs(u.get(k,0)-v.get(k,0)) for k in u.keys()|v.keys()))
   metrics[name]=max(errors)
  values.append(dict(key=key,max_weight_drift=metrics))
 report=dict(schema='xgpu.dp.distance.diagnostic.v1',left_heads=len(left),right_heads=len(right),paired_heads=len(values),
     per_head=values,max_weight_drift={k:max(r['max_weight_drift'][k] for r in values) for k in values[0]['max_weight_drift']} if values else {},
     max_cpu32_reconstruction_error=max((r['native32_vs_recorded_d2'] for r in left+right),default=None),
     max_float64_vs_stable_error=max((r['float64_vs_stable_d2'] for r in left+right),default=None),
     max_stored_norm_residual=max((r['stored_norm_residual_max'] for r in left+right),default=None),
     formal_recipe_changed=False,scope='same-card first-head diagnostic; cross-card inference requires captured features from both cards',
     limitation='plain Euclidean distance discards an artifact rounding residual and therefore changes its metric; not a formal recipe repair')
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='per_head'}))


if __name__=='__main__':main()
