"""Extract original recorded paired LIBERO views; never resize or invent a camera."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path

from .assets import write_json


def extract(source, output, language):
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = [json.loads(line) for line in (source/'meta/episodes.jsonl').read_text().splitlines()]
    ep = next(row for row in episodes if language in row['tasks'] and row['length'] >= 3)
    info = json.loads((source/'meta/info.json').read_text())
    index = ep['episode_index']
    parquet = source/info['data_path'].format(episode_chunk=index//info['chunks_size'], episode_index=index)
    rows = pq.ParquetFile(parquet).read_row_group(0, columns=['image','wrist_image','state','frame_index','task_index'], use_threads=False).slice(0,3).to_pylist()
    pairs, provenance = [], []
    for row in rows:
        cameras, hashes = {}, {}
        for key in ('image','wrist_image'):
            encoded = row[key]['bytes']
            pixels = np.asarray(Image.open(io.BytesIO(encoded)).convert('RGB')).copy()
            if pixels.shape != (256,256,3) or pixels.dtype != np.uint8:
                raise ValueError('recorded original LIBERO camera must be RGB256: '+key)
            cameras[key] = pixels
            hashes[key] = dict(encoded_sha256=hashlib.sha256(encoded).hexdigest(),
                               rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        pairs.append(cameras)
        provenance.append(dict(episode_index=index, frame_index=row['frame_index'],
                               dataset_task_index=row['task_index'], cameras=hashes))
    states = np.asarray([row['state'] for row in rows], dtype=np.float32)
    assert states.shape == (3,8) and np.isfinite(states).all()
    prompts = np.asarray([language]*3)
    images = np.stack([pair['image'] for pair in pairs])
    wrists = np.stack([pair['wrist_image'] for pair in pairs])
    np.savez_compressed(output/'wire_observations.npz', **{'observation/image':images,
        'observation/wrist_image':wrists,'observation/state':states,'prompt':prompts})
    np.savez_compressed(output/'fastwam_observations.npz', image=images, image2=wrists, state=states, prompt=prompts)
    receipt = dict(schema='r17.Q.recorded_observations.v1', synthetic=False, recorded=True,
        real_reset=False, quality_claim=False, rows=3, image_shape=list(images.shape),
        wrist_shape=list(wrists.shape), state_shape=list(states.shape), language=language,
        source=str(parquet), source_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
        dataset_info=info, frames=provenance)
    receipt['files'] = {name:hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ('wire_observations.npz','fastwam_observations.npz')}
    write_json(output/'OBSERVATIONS.json', receipt)
    return receipt


def validate(directory):
    """CPU validation independent of parquet/PIL; usable on both model islands."""
    import numpy as np
    directory = Path(directory)
    r = json.loads((directory/'OBSERVATIONS.json').read_text())
    assert r['synthetic'] is False and r['recorded'] is True and r['rows']==3
    for name, digest in r['files'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest, name+' changed'
    with np.load(directory/'wire_observations.npz',allow_pickle=False) as wire, np.load(directory/'fastwam_observations.npz',allow_pickle=False) as fast:
        for key,native,original in (('observation/image','image','image'),('observation/wrist_image','image2','wrist_image')):
            assert wire[key].shape==(3,256,256,3) and wire[key].dtype==np.uint8
            assert np.array_equal(wire[key],fast[native])
            for i,frame in enumerate(r['frames']):
                assert hashlib.sha256(wire[key][i].tobytes()).hexdigest()==frame['cameras'][original]['rgb_sha256']
        assert wire['observation/state'].shape==(3,8) and np.isfinite(wire['observation/state']).all()
        assert np.array_equal(wire['observation/state'],fast['state'])
        assert np.array_equal(wire['prompt'],fast['prompt']) and all(p==r['language'] for p in wire['prompt'])
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--language',required=True)
    a=p.parse_args();extract(a.source,a.output,a.language);validate(a.output);print('PASS original paired RGB256 recorded LIBERO frames')
