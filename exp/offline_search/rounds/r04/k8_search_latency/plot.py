"""Standalone summary figure; bands span the two fresh-process p50 values."""
import os;os.sched_setaffinity(0,{35});os.environ['MPLCONFIGDIR']='/tmp/k8_latency/mpl'
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for model,color in [('pi05','#2464a4'),('groot','#c06b22')]:
 for suite,style in [('l10','-'),('spatial','--')]:
  x=[];lo=[];hi=[];med=[]
  for scale,factor in ((50,1),(500,1),(500,2),(500,4),(500,10)):
   rr=[]
   for rep in (1,2):
    rows=json.loads((OUT/f'scaling_r{rep}.json').read_text());rr.append(next(r['ms']['p50'] for r in rows if r['config']==f'{model}_{suite}_{scale}_AWM' and r['factor']==factor))
   x.append(scale*factor);med.append(np.mean(rr));lo.append(min(rr));hi.append(max(rr))
  axes[0].plot(x,med,style,marker='o',color=color,label=f'{model} {suite}');axes[0].fill_between(x,lo,hi,color=color,alpha=.12)
axes[0].set(xscale='log',xlabel='Episodes per suite (1k–5k: tiled 500 fit)',ylabel='Median query latency (ms)',title='AWM: fixed task and increasing candidate block')
for name,color in [('pi05_l10_50_AWM','#70a2d0'),('pi05_l10_500_AWM','#2464a4'),('pi05_l10_50_MixedJudge','#df9870'),('pi05_l10_500_MixedJudge','#a44920')]:
 rr=[json.loads((OUT/f'concurrency_{name}_r{rep}.json').read_text()) for rep in (1,2)]
 x=[r['threads'] for r in rr[0]];v=np.asarray([[r['ms']['p50'] for r in rows] for rows in rr])
 axes[1].plot(x,v.mean(0),marker='o',color=color,label=name.replace('pi05_l10_','').replace('_',' '));axes[1].fill_between(x,v.min(0),v.max(0),color=color,alpha=.12)
axes[1].set(xlabel='Threads in one process (4 physical cores)',ylabel='Median query latency (ms)',title='π0.5 l10: same 1,024 calls at every thread count',xticks=[1,4,8,16,24,32])
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
fig.suptitle('K8 CPU search latency · shading = repeat range, not confidence interval',fontsize=11)
fig.savefig(OUT/'latency.png',dpi=160);fig.savefig(OUT/'latency.svg');plt.close(fig)
