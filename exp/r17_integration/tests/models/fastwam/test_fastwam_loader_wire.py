from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.r17_integration.models.fastwam.loader import LoadSpec, audit_parameters, inspect_config, read_stats, safetensor_header
from exp.r17_integration.models.fastwam.manifest import content_identity
from exp.r17_integration.tests.models.fastwam.conftest import FIXTURES, make_adapter, native_defs, observation, request


def _base(tmp_path):
    root=tmp_path/"released_libero";root.mkdir()
    config=json.loads((FIXTURES/"libero_config.json").read_text())
    (root/"config.json").write_text(json.dumps(config))
    return root,config


def test_config_embodiments_and_delta_rejection(tmp_path):
    root,config=_base(tmp_path)
    cfg=inspect_config(LoadSpec("libero",str(root),str(tmp_path)))
    assert cfg==config and cfg["action_dim"]==7 and cfg["proprio_dim"]==8
    delta=tmp_path/"immutable_delta";delta.mkdir()
    (delta/"mt50.json").write_text(json.dumps({"format":"fastwam-mt50-delta-v1","mode":"frozen_video"}))
    mw=inspect_config(LoadSpec("metaworld",str(delta),str(tmp_path),str(root)))
    assert (mw["action_dim"],mw["proprio_dim"],mw["image_size"],mw["toggle_action_dimensions"])==(4,4,[224,224],[])
    assert mw["action_dit_config"]["action_dim"]==4
    with pytest.raises(ValueError):
        inspect_config(LoadSpec("libero",str(delta),str(tmp_path)))
    config["action_dim"]=4;(root/"config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        inspect_config(LoadSpec("libero",str(root),str(tmp_path)))
    with pytest.raises(ValueError):
        LoadSpec("metaworld",str(tmp_path/"policy_latest"),str(tmp_path))
    with pytest.raises(ValueError):
        audit_parameters({"head":[4,8],"state":[8,4]}, {"head":[4,8]})
    with pytest.raises(ValueError):
        audit_parameters({"head":[4,8]}, {"head":[7,8]})


def test_safetensors_header_strict_small_stats_and_content_hash(tmp_path):
    from safetensors.numpy import save_file
    root,config=_base(tmp_path)
    arrays={name+"."+bound:np.arange(d,dtype=np.float32)+(2 if bound=="max" else 0)
        for name,d in (("action",7),("observation.state",8)) for bound in ("min","max")}
    for name in ("policy_preprocessor_step_3_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        save_file(arrays,str(root/name))
    stats=read_stats(LoadSpec("libero",str(root),str(tmp_path)))
    assert len(stats["action"]["min"])==7
    assert safetensor_header(root/"policy_preprocessor_step_3_normalizer_processor.safetensors")["action.min"]["shape"]==[7]
    before=content_identity({"base":root/"config.json"})
    (root/"config.json").write_text((root/"config.json").read_text()+" ")
    assert before!=content_identity({"base":root/"config.json"})
    arrays["action.min"][0]+=1
    save_file(arrays,str(root/"policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    with pytest.raises(ValueError):
        read_stats(LoadSpec("libero",str(root),str(tmp_path)))


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_wire_parity_original_resize_minmax_toggle(benchmark):
    a=make_adapter(benchmark);obs=observation(benchmark)
    obs["image"]=np.arange(obs["image"].size,dtype=np.uint8).reshape(obs["image"].shape)
    p=a.prepare(obs,request())
    if benchmark=="metaworld":
        from exp.mw_finetune.fastwam.common import resize_rgb,normalize,denormalize
        expected=torch.from_numpy(resize_rgb(obs["image"]).copy()).permute(2,0,1).unsqueeze(0).float()/255
        np.testing.assert_array_equal(p.data["proprio"].numpy()[0],normalize(obs["state"],a.wire.stats["observation.state"]))
    else:
        defs=native_defs("modeling_fastwam_native.py.txt",{"Tensor":torch.Tensor})
        batch={"observation.images."+name:torch.from_numpy(obs[name].copy()).permute(2,0,1).unsqueeze(0).float()/255
            for name in ("image","image2")}
        expected=defs["_input_image_from_batch"](batch,SimpleNamespace(image_size=[224,448],model_video_frames=9))
    torch.testing.assert_close(p.data["input_image"],expected,rtol=0,atol=0)
    x=np.tile(np.linspace(-2,2,a.metadata().layout.action_width,dtype=np.float32),(32,1))
    wire=a.to_wire(torch.from_numpy(x),p).actions
    low,high=a.wire.stats["action"]["min"],a.wire.stats["action"]["max"]
    if benchmark=="metaworld":
        ref=denormalize(x,a.wire.stats["action"])
    else:
        ref=((torch.from_numpy(x)+1)/2*(torch.from_numpy(high)-torch.from_numpy(low))+torch.from_numpy(low))
        source=(FIXTURES/"processor_fastwam_native.py.txt").read_text()
        import ast
        tree=ast.parse(source);cls=tree.body[0]
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="action")
        module=ast.Module(body=[method],type_ignores=[])
        ns={"PolicyAction":torch.Tensor,"torch":torch}
        exec(compile(module,"vendor_toggle.action","exec"),ns)
        ref=ns["action"](SimpleNamespace(toggle_dimensions=[-1]),ref).numpy()
    assert wire.tobytes()==ref.astype(np.float32).tobytes()
    assert not np.array_equal(wire[:,:-1],np.clip(wire[:,:-1],-1,1))
    del obs["image"]
    with pytest.raises(ValueError):
        a.prepare(obs,request())


def test_import_default_off_no_torch_vendor_jax_or_registration():
    source="""
import sys
import exp.r17_integration.models.fastwam
from exp.r17_integration.common.provider import get_provider
assert not any(n in sys.modules for n in ['torch','jax','lerobot','fastwam'])
try: get_provider('fastwam_mt50')
except ValueError: pass
else: raise AssertionError('provider implicitly registered')
"""
    result=subprocess.run([sys.executable,"-c",source],text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_frozen_legacy_and_training_files_byte_identical():
    from exp.r17_integration.models.fastwam.manifest import sha256_file
    before=json.loads((Path(__file__).parent/"receipts/legacy_before.json").read_text())
    for path,sha in before.items():
        if path.startswith("src/"):
            continue  # P1 owns additive shared modifications; legacy tests run separately.
        assert sha256_file(path)==sha,path


def test_libero_original_processor_zero_tiny_span_and_toggle_midpoint():
    from enum import Enum
    Mode=Enum("Mode","IDENTITY MEAN_STD MIN_MAX QUANTILES QUANTILE10")
    defs=native_defs("normalizer_native.py.txt",{"Tensor":torch.Tensor,"FeatureType":str,"NormalizationMode":Mode})
    a=make_adapter("libero")
    # MIN_MAX in LIBERO replaces only zero denominators; C clamps spans at1e-6.
    a.wire.stats["observation.state"]["max"][:2]=[a.wire.stats["observation.state"]["min"][0],
        a.wire.stats["observation.state"]["min"][1]+np.float32(1e-7)]
    obs=observation("libero");obs["state"][:2]=[-1.,-1.]
    p=a.prepare(obs,request())
    stats={k:torch.from_numpy(v.copy()) for k,v in a.wire.stats["observation.state"].items()}
    fake=SimpleNamespace(norm_map={"STATE":Mode.MIN_MAX},_tensor_stats={"state":stats},eps=1e-8)
    expected=defs["_apply_transform"](fake,torch.from_numpy(obs["state"]),"state","STATE")
    assert p.data["proprio"].numpy()[0].tobytes()==expected.numpy().tobytes()
    x=np.zeros((32,7),np.float32)
    x[:3,-1]=[-1.,0.,1.]
    # After minmax inverse to[0,1], published toggle preserves midpoint zero.
    np.testing.assert_array_equal(a.to_wire(x,None).actions[:3,-1],[1.,0.,-1.])


@pytest.mark.parametrize("benchmark",["libero","metaworld"])
def test_canonical_benchmark_aliases(benchmark):
    a=make_adapter(benchmark);raw=observation(benchmark)
    canonical={"observation/image":raw["image"],"observation/state":raw["state"],"prompt":raw["prompt"]}
    if benchmark=="libero":
        canonical["observation/wrist_image"]=raw["image2"]
    p1,p2=(a.prepare(obs,request()) for obs in (raw,canonical))
    torch.testing.assert_close(p1.data["input_image"],p2.data["input_image"],rtol=0,atol=0)
    np.testing.assert_array_equal(p1.state,p2.state)
    with pytest.raises(ValueError):
        a.prepare({**canonical,"state":raw["state"]+1},request())


def test_real_published_small_config_norm_and_header_probe():
    from exp.r17_integration.models.fastwam.loader import audit_released_header
    root=Path("/data/mw_finetune/models/fastwam_libero")
    if not root.is_dir():
        pytest.skip("published local LIBERO config/norm/header not installed")
    spec=LoadSpec("libero",str(root),"/data/xwam/wan22_5b")
    config=inspect_config(spec);stats=read_stats(spec);header=audit_released_header(spec)
    assert config["action_dim"]==7 and config["proprio_dim"]==8
    assert config["image_size"]==[224,448] and config["action_horizon"]==32
    assert len(header)==1651
    assert len(stats["action"]["min"])==7 and len(stats["observation.state"]["min"])==8


def test_exact_c_delta_loader_rejects_missing_extra_and_wrong_tensors(tmp_path):
    from exp.mw_finetune.fastwam.model import canonical_key,load_delta
    from safetensors.torch import save_file
    # Execute C's real strict loader with tiny trainable parameter views.
    # This tests exact loading semantics without constructing its 6B network.
    trainable={"model.action_expert.head.weight":torch.nn.Parameter(torch.zeros(4,8)),
        "model.proprio_encoder.weight":torch.nn.Parameter(torch.zeros(8,4))}
    frozen=torch.nn.Parameter(torch.zeros(2,2),requires_grad=False)
    policy=SimpleNamespace(named_parameters=lambda:list(trainable.items())+[("model.video_expert.weight",frozen)])
    good={canonical_key(k):torch.arange(p.numel(),dtype=torch.float32).reshape(p.shape) for k,p in trainable.items()}
    file=tmp_path/"action.safetensors";save_file(good,str(file))
    load_delta(policy,tmp_path)
    for name,p in trainable.items():
        torch.testing.assert_close(p,good[canonical_key(name)],rtol=0,atol=0)
    assert torch.equal(frozen,torch.zeros_like(frozen))
    bads=[dict(list(good.items())[:1]),{**good,"unexpected":torch.zeros(1)},
          {**good,canonical_key("model.proprio_encoder.weight"):torch.zeros(8,7)}]
    for bad in bads:
        save_file(bad,str(file))
        with pytest.raises(ValueError):
            load_delta(policy,tmp_path)
