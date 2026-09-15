import pickle,struct,collections,pathlib,json,time,gc,resource
class Record:
    def __new__(cls,*args,**kwargs): return object.__new__(cls)
    def __init__(self,*args,**kwargs): pass
    def __setstate__(self,state):
        if isinstance(state,dict): self.__dict__.update(state)
        elif isinstance(state,tuple):
            for part in state:
                if isinstance(part,dict): self.__dict__.update(part)
class ArrayStub:
    def __new__(cls,*args,**kwargs): return object.__new__(cls)
    def __init__(self,*args,**kwargs): pass
    def __setstate__(self,state): pass
def identity(value): return value
class MetadataUnpickler(pickle._Unpickler):
    dispatch=pickle._Unpickler.dispatch.copy()
    def find_class(self,module,name):
        if module.startswith("numpy"): return ArrayStub
        if module=="openpi.cache.storage_types" and name in ("CacheEntry","CachePayload"): return Record
        if module=="openpi.cache.types" and name=="CheckpointID": return identity
        if module=="collections" and name=="OrderedDict": return collections.OrderedDict
        if module.startswith("openpi."): return Record
        return ArrayStub
    def load_binbytes(self):
        n=struct.unpack("<I",self.read(4))[0]
        self.read(n)
        self.append(b"")
    def load_binbytes8(self):
        n=struct.unpack("<Q",self.read(8))[0]
        self.read(n)
        self.append(b"")
    def load_bytearray8(self):
        n=struct.unpack("<Q",self.read(8))[0]
        self.read(n)
        self.append(bytearray())
    dispatch[pickle.BINBYTES[0]]=load_binbytes
    dispatch[pickle.BINBYTES8[0]]=load_binbytes8
    dispatch[pickle.BYTEARRAY8[0]]=load_bytearray8
for suite in ("libero_spatial","libero_10"):
    p=pathlib.Path("/data/openpi/ablation_study/cache_size/artifacts")/("cache_size_"+suite+"_all_S6.pkl")
    start=time.time()
    with p.open("rb") as f: artifact=MetadataUnpickler(f).load()
    groups=collections.defaultdict(list)
    for e in artifact["entries"]: groups[e.trajectory_id].append(e)
    rows=[]
    for tid,entries in sorted(groups.items()):
        steps=sorted(e.step_idx for e in entries)
        assert all(isinstance(v,int) for v in steps),(tid,"non-integer step")
        assert len(set(steps))==len(steps),(tid,"duplicate steps")
        task_keys=sorted(set(e.payload.task_key for e in entries))
        rows.append(dict(trajectory_id=tid,task_id=int(tid.split("/")[0].split("_")[-1]),task_keys=task_keys,length=len(entries),first_step=steps[0],last_step=steps[-1],step_increments=sorted(set(b-a for a,b in zip(steps,steps[1:])))))
    print(json.dumps(dict(suite=suite,path=str(p),file_bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,entries=len(artifact["entries"]),n_trajectories=len(rows),elapsed_s=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rows=rows)),flush=True)
    del artifact,groups,rows,entries
    gc.collect()

