import os,time,json,pathlib,numpy as np
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def fps(x,k=512):
 d=np.full(len(x),np.inf,dtype=np.float32);idx=0;out=[]
 for _ in range(k):
  out.append(idx);z=x-x[idx];d=np.minimum(d,np.einsum('ij,ij->i',z,z));idx=int(np.argmax(d))
 return np.asarray(out)
rng=np.random.default_rng(7);rows=[]
for n in [1024,4096,16384]:
 x=rng.normal(size=(n,3)).astype(np.float32);fps(x);times=[]
 for _ in range(8):
  t=time.perf_counter();fps(x);times.append(time.perf_counter()-t)
 rows.append(dict(input_points=n,output_points=512,mean_s=float(np.mean(times)),p95_s=float(np.percentile(times,95))))
p=pathlib.Path(__file__).with_name('fps_cpu_results.json');p.write_text(json.dumps(dict(note='synthetic points, pure numpy exact greedy FPS, start index 0; not full DP3 frontend',rows=rows),indent=2));print(p.read_text())
