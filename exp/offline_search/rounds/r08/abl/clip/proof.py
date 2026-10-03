"""Portable encoder comparison; --reference writes exact images and open_clip outputs."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

if __package__:
    from .encoder import Encoder, preprocess, weights_path, RECIPE, WEIGHT_SHA256
else:
    from encoder import Encoder, preprocess, weights_path, RECIPE, WEIGHT_SHA256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',action='store_true')
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if a.reference:
        import open_clip
        from PIL import Image
        root=Path('/home/weiland/trace_runs/offline_search_store/library/pi05_l10/current/tok')
        imgs=[np.array(np.load(root/f'img{i}.npy',mmap_mode='r')[r]) for r in (0,53) for i in (0,1)]
        imgs += [np.random.default_rng(18).integers(0,256,(256,256,3),dtype=np.uint8),
                 np.random.default_rng(19).integers(0,256,(129,391,3),dtype=np.uint8)]
        ref,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained=str(weights_path()),force_quick_gelu=True)
        ref.eval()
        tensor=torch.stack([pre(Image.fromarray(im)) for im in imgs])
        pre_diff=max(float((pre(Image.fromarray(im))-preprocess(im)).abs().max()) for im in imgs)
        with torch.inference_mode():
            expected=ref.encode_image(tensor,normalize=True).numpy()
        np.savez(a.inputs,**{f'image{i}':im for i,im in enumerate(imgs)},reference=expected,
                 preprocessed=tensor.numpy())
    else:
        z=np.load(a.inputs,allow_pickle=False)
        imgs=[z[f'image{i}'] for i in range(len(z.files)-2)]
        expected=z['reference']
        pre_diff=float(np.max(np.abs(np.stack([preprocess(im).numpy() for im in imgs])-z['preprocessed'])))
    t=time.perf_counter()
    enc=Encoder('cpu')
    got=enc.encode(imgs)
    diff=float(np.max(np.abs(got-expected)))
    result={'PASS':pre_diff==0 and diff<1e-5,'images':len(imgs),'preprocess_max_abs_diff':pre_diff,
            'embedding_max_abs_diff':diff,'bit_equal':bool(np.array_equal(got,expected)),
            'torch':torch.__version__,'python':platform.python_version(),'host':platform.node(),
            'recipe':RECIPE,'weight_sha256':WEIGHT_SHA256,'device':'cpu','wall_s':time.perf_counter()-t}
    a.output.write_text(json.dumps(result,indent=1)+'\n')
    print(json.dumps(result),flush=True)
    assert result['PASS']


if __name__=='__main__':
    main()
