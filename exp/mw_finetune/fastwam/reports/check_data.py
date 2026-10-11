from exp.mw_finetune.fastwam.data import Dataset
import torch
x=Dataset('/data/mw_finetune/fastwam/smoke_224',allow_partial=True)
a=x[0]
print({k:(tuple(v.shape),str(v.dtype)) if isinstance(v,torch.Tensor) else v for k,v in a.items()})
print('image range',a['video'].min().item(),a['video'].max().item())
print('state',a['proprio'])
