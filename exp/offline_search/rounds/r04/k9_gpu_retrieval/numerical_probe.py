"""CPU-only diagnosis of the actual early-distance cancellation and boundary ties."""
from common import *
result=[]
for suite,scale,task in [('l10',500,4),('l10',500,5),('l10',50,5),('spatial',500,9)]:
 cfg=next(c for c in configs() if c['id']==f'pi05_{suite}_{scale}_AWM')
 m=load(cfg['path']);seq=sequences(suite,episodes_per_task=2,max_steps=1)
 for ep,qs in seq:
  if ep.task_id!=task:continue
  q=qs[0];T,step,regime,k0,k1,xv,rs8,d,med,c,dt=m._dist(q)
  x=np.r_[xv,rs8];y=x@T.W0f-T.c0;cross=T.Z@(T.A0@y)
  d2=T.n20-2*cross+float(y@y)
  # Cast intermediates at progressively earlier boundaries; not ground truth.
  x64=np.r_[m.B0T.astype('f8')@k0-m.muB0,m.B1T.astype('f8')@k1-m.muB1,rs8]
  y64=x64@T.W0f.astype('f8')-T.c0
  c64=T.Z.astype('f8')@(T.A0.astype('f8')@y64)
  d264=T.n20.astype('f8')-2*c64+y64@y64
  chosen=np.argsort(dt,kind='stable')[:18]
  terms=np.abs(T.n20.astype('f8'))+2*np.abs(cross.astype('f8'))+abs(float(y@y))
  result.append(dict(cell=cfg['id'],uid=ep.uid,regime=regime,
      pca_f32_vs_f64_max_abs=float(np.abs(x64[:128]-xv).max()),
      max_distance_f32_vs_f64=float(np.abs(d-np.sqrt(np.maximum(d264,0))).max()),
      selection=[dict(row=int(T.rows[i]),cpu_d=float(d[i]),cpu_d2=float(d2[i]),f64_d=float(np.sqrt(max(d264[i],0))),
                      n20=float(T.n20[i]),twice_cross=float(2*cross[i]),query_norm2=float(y@y),
                      cancellation_ratio=float(terms[i]/max(abs(float(d2[i])),1e-12))) for i in chosen]))
dump(OUT/'numerical_probe.json',result)
print('Saved',len(result),'real step-0 probes')
