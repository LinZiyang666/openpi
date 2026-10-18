"""Observe existing first-head locals without repeating or changing model/math calls.

Diagnostic timing arm only. The native return object, branches, weights and
recipe remain unchanged. Captures stage1 features, actual x/y and raw d2.
"""
from dataclasses import asdict,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import numpy as np


def install(root):
 from exp.r17_integration.common.layers.recipe import LayoutRecipe
 original=LayoutRecipe._retrieve;root=Path(root);root.mkdir(parents=True,exist_ok=True)
 lock=threading.Lock();count=0
 def retrieve(self,q,*args,**kwargs):
  nonlocal count
  if q.step!=0 or getattr(q,'request',None) is None or self.tasks[q.task_id].early is None:
   return original(self,q,*args,**kwargs)
  request=asdict(q.request) if is_dataclass(q.request) else dict(vars(q.request))
  # Exclude recipe fit/qualification queries; these are actual formal episodes.
  if request.get('decision_seq')!=0:return original(self,q,*args,**kwargs)
  previous=sys.getprofile();captured={}
  def profile(frame,event,arg):
   if event=='return' and frame.f_code is original.__code__:captured.update(frame.f_locals)
   if previous:previous(frame,event,arg)
  sys.setprofile(profile)
  try:result=original(self,q,*args,**kwargs)
  finally:sys.setprofile(previous)
  needed=('x','y','d2','chosen','pos','wn','distances')
  if any(k not in captured for k in needed):raise ValueError('native early distance locals missing')
  T=self.tasks[q.task_id];arrays={k:np.asarray(captured[k]).copy() for k in needed}
  arrays.update(Z=T.Z.copy(),A0=T.early.A0.copy(),n20=T.early.n20.copy(),
      W0f=T.early.W0f.copy(),c0=T.early.c0.copy(),rows=T.rows.copy(),state=np.asarray(q.state).copy())
  for i,name in enumerate(self.fields):
   arrays['feature_'+str(i)]=np.asarray(q.features[name]).copy()
   arrays['pca_B_'+str(i)]=self.pca[name][0].copy();arrays['pca_shift_'+str(i)]=self.pca[name][1].copy()
  arrays['valid_state_indices']=np.asarray(self.layout.valid_state_indices,dtype=np.int64)
  with lock:
   index=count;count+=1
   stem=root/(str(os.getpid())+'_'+str(index))
   np.savez_compressed(stem.with_suffix('.npz'),**arrays)
   with stem.with_suffix('.npz').open('rb') as f:pin=hashlib.file_digest(f,'sha256').hexdigest()
   stem.with_suffix('.json').write_text(json.dumps(dict(request=request,task_id=int(q.task_id),
       feature_fields=list(self.fields),k=int(self.k),kref=int(self.kref),npz_sha256=pin,
       scope='native formal first-head locals; output unchanged; observer may affect scheduling'),sort_keys=True)+'\n')
  return result
 LayoutRecipe._retrieve=retrieve
