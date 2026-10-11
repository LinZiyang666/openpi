"""Compare actual observation bytes and channel differences, without tolerance claims."""
import argparse, hashlib, json, pathlib
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--reference',type=pathlib.Path,required=True);ap.add_argument('--target',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
ref=np.load(args.reference/'worker0/observations.npz'); ref_doc=json.loads((args.reference/'worker0/RESULT.json').read_text())
workers=[]
for worker in sorted(args.target.glob('worker*')):
 target=np.load(worker/'observations.npz'); doc=json.loads((worker/'RESULT.json').read_text())
 if set(ref.files)!=set(target.files): raise RuntimeError('observation key set differs')
 if [(e['suite'],e['task_id'],e['original_init_idx'],e['state_sha256'],e['prompt']) for e in ref_doc['episodes']] != [(e['suite'],e['task_id'],e['original_init_idx'],e['state_sha256'],e['prompt']) for e in doc['episodes']]: raise RuntimeError('fixed A state/task bytes differ')
 rows=[]; groups={}
 for k in sorted(ref.files):
  a,b=ref[k],target[k]
  if a.shape!=b.shape or a.dtype!=b.dtype: raise RuntimeError('shape/dtype differs '+k)
  delta=np.abs(a.astype(np.float64)-b.astype(np.float64)); mismatched=np.count_nonzero(delta)
  row={'key':k,'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'exact_bytes':a.tobytes()==b.tobytes(),'changed_elements':int(mismatched),'elements':int(a.size),'max_abs':float(delta.max()),'mae':float(delta.mean()),'reference_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'target_sha256':hashlib.sha256(b.tobytes()).hexdigest()}; rows.append(row)
  group=('wire'+k.rsplit('_wire',1)[1] if '_wire' in k else 'raw_pixels') if a.dtype==np.uint8 else 'state_vectors'
  g=groups.setdefault(group,{'arrays':0,'exact_arrays':0,'elements':0,'changed_elements':0,'abs_sum':0.0,'square_sum':0.0,'max_abs':0.0})
  g['arrays']+=1; g['exact_arrays']+=row['exact_bytes'];g['elements']+=a.size;g['changed_elements']+=mismatched;g['abs_sum']+=float(delta.sum());g['square_sum']+=float(np.square(delta).sum());g['max_abs']=max(g['max_abs'],row['max_abs'])
 for g in groups.values():
  g['elements']=int(g['elements']);g['changed_elements']=int(g['changed_elements']);g['mae']=g.pop('abs_sum')/g['elements'];g['rmse']=(g.pop('square_sum')/g['elements'])**.5;g['changed_fraction']=g['changed_elements']/g['elements']
 workers.append({'worker':worker.name,'fixed_A_state_bytes_equal':True,'groups':groups,'arrays':rows})
base=np.load(args.target/'worker0/observations.npz'); intra=[]
for worker in sorted(args.target.glob('worker*'))[1:]:
 data=np.load(worker/'observations.npz'); changed=[k for k in base.files if base[k].tobytes()!=data[k].tobytes()];intra.append({'worker':worker.name,'exact_arrays':len(base.files)-len(changed),'total_arrays':len(base.files),'changed_keys':changed})
result={'schema':'r17.timan1_pixel_comparison.v1','status':'PASS','reference':str(args.reference),'target':str(args.target),'cases':ref_doc['episodes'],'workers':workers,'within_timan1':intra}
args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':result['status'],'groups':workers[0]['groups'],'within_timan1':intra},indent=2))
