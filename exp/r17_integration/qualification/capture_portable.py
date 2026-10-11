"""Q boundary serialization for BF16, retaining exact original uint16 bits.

P6 v1 calls Tensor.numpy(), which cannot encode BF16. This shim copies only
observer products; inference tensors/RNG and native resume objects stay untouched.
"""
from __future__ import annotations
from dataclasses import replace


class PortableSink:
    def __init__(self,sink): self.sink=sink
    def on_execution(self,ack): return self.sink.on_execution(ack)
    def __call__(self,capture):
        import torch
        bits={}
        def convert(value,path):
            if isinstance(value,torch.Tensor) and value.dtype==torch.bfloat16:
                cpu=value.detach().cpu().contiguous()
                bits[path]=cpu.view(torch.uint16).numpy().copy()
                return cpu.float().numpy().copy()
            if isinstance(value,dict): return {k:convert(v,path+'/'+str(k)) for k,v in value.items()}
            if isinstance(value,tuple): return tuple(convert(v,path+'/'+str(i)) for i,v in enumerate(value))
            if isinstance(value,list): return [convert(v,path+'/'+str(i)) for i,v in enumerate(value)]
            return value
        snapshots=tuple(replace(s,x_t=convert(s.x_t,'snapshot/'+str(s.loop_index)),
            noise_state=convert(s.noise_state,'snapshot_rng/'+str(s.loop_index))) for s in capture.native_snapshots)
        product=convert(dict(capture.model_capture),'model_capture')
        normalized=convert(capture.normalized_chunk,'normalized_chunk')
        fields=convert(dict(capture.fields),'fields')
        if bits:
            product['q_original_bfloat16_bits']=bits
            product['q_bfloat16_serialization']={'status':'exact_uint16_bits_plus_lossless_float32_values',
                'native_dtype':'bfloat16','producer':'Q observer boundary; upstream request Q_to_P6'}
        return self.sink(replace(capture,normalized_chunk=normalized,fields=fields,
            native_snapshots=snapshots,model_capture=product))
