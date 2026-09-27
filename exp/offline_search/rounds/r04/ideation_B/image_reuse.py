"""Exact same-image opportunities, uint8 image differences, same-episode adjacent tok rows.
Exact memoization does not change any vision representation. No approximate reuse policy is evaluated.
"""
import json,pathlib,concurrent.futures,numpy as np
OUT=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/weiland/trace_runs/offline_search_store')
def run(cell):
 d=ROOT/'queries'/cell
 rows=np.load(d/'tok/rows.npy');step=np.load(d/'step.npy',mmap_mode='r');ep=np.load(d/'ep.npy',mmap_mode='r')
 good=(np.diff(rows)==1)&(np.diff(ep[rows])==0)
 valid=np.flatnonzero(good)+1;ims=[np.load(d/f'tok/img{i}.npy',mmap_mode='r') for i in range(2)]
 eq=[];ma=[]
 for im in ims:
  ee=[];mm=[]
  for lo in range(0,len(valid),64):
   ix=valid[lo:lo+64];delta=np.abs(im[ix].astype(np.int16)-im[ix-1].astype(np.int16))
   ee.extend((delta==0).all((1,2,3)));mm.extend(delta.mean((1,2,3)))
  eq.append(np.array(ee));ma.append(np.array(mm))
 return {'cell':cell,'tok_rows':len(rows),'adjacent_pairs':len(valid),'exact_camera0':float(np.mean(eq[0])),'exact_camera1':float(np.mean(eq[1])),'exact_both':float(np.mean(eq[0]&eq[1])),'camera_MAE_quantiles': [np.quantile(m,[0,.1,.5,.9,1]).tolist() for m in ma],'exact_camera_pairs':int(eq[0].sum()+eq[1].sum()),'pair_key_compute_saved_fraction_two_active_cameras':float((eq[0].sum()+eq[1].sum())/(2*len(rows)))}
if __name__=='__main__':
 cells=[f'{m}_{s}_{a}' for m in ['pi05','groot'] for s in ['spatial','l10'] for a in ['inf','cache']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:out=list(ex.map(run,cells))
 (OUT/'image_reuse.json').write_text(json.dumps(out,indent=2))
 for x in out:print(x)
