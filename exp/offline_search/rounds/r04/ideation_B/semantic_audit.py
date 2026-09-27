"""CPU symbolic mask check and source-derived transformer cost arithmetic.
No policy inference; validates the proposed masked-token packing's mask/position algebra only.
"""
import json,pathlib,numpy as np
O=pathlib.Path(__file__).resolve().parent
# Three 256-token camera slots, 200 language slots; two active cameras.
checks=[]
for valid_text in [1,20,50,100,200]:
 pad=np.r_[np.ones(512,bool),np.zeros(256,bool),np.arange(200)<valid_text]
 keep=np.r_[np.arange(512),np.arange(768,968)] # drop ONLY the fixed dummy camera, leave text padding stable
 ppos=np.cumsum(pad)-1
 am=pad[:,None]&pad[None,:]
 packed_pad=pad[keep];newpos=np.cumsum(packed_pad)-1;newmask=packed_pad[:,None]&packed_pad[None,:]
 checks.append({'text_valid':valid_text,'positions_equal':bool(np.array_equal(ppos[keep],newpos)),'prefix_mask_equal':bool(np.array_equal(am[np.ix_(keep,keep)],newmask)),'suffix_position_offset_equal':bool(pad.sum()==packed_pad.sum())})
# SigLIP GEMM MAC count: QKV+output 4ND^2, MLP 2NDM, attention 2N^2D.
d,m=1152,4304
mac=lambda n:4*n*d*d+2*n*d*m+2*n*n*d
vision=[]
for depth,res in [(27,224),(6,224),(12,224),(18,224),(27,168),(27,112)]:
 n=(res//14)**2
 vision.append({'depth':depth,'resolution':res,'patches':n,'transformer_MAC_ratio_vs27_224':depth*mac(n)/(27*mac(256)),'note':'analytical only, not latency; excludes stem, pooling, projector, launch/memory costs'})
obj={'mask_checks':checks,'prefix_tokens_before':968,'prefix_tokens_after_dummy_drop':712,'linear_token_work_ratio':712/968,'attention_quadratic_work_ratio':(712/968)**2,'vision_compute':vision}
(O/'semantic_audit.json').write_text(json.dumps(obj,indent=2));print(json.dumps(obj,indent=2))
