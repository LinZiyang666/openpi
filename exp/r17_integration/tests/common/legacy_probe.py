"""Subprocess certificate comparing default-off behavior to checked-in HEAD."""
import asyncio
from dataclasses import asdict
import hashlib
import json
import importlib
import subprocess
import sys
import types


if "--baseline" in sys.argv:
    # Only read git objects; never write or swap the shared checkout.
    for name in ("openpi.cache.types", "openpi.cache.storage_types", "openpi.cache.config", "openpi.serving.websocket_policy_server"):
        path="src/"+name.replace(".","/")+".py"
        source=subprocess.check_output(["git","show","HEAD:"+path],text=True)
        module=types.ModuleType(name)
        module.__file__=path
        module.__package__=name.rsplit(".",1)[0]
        parent=importlib.import_module(module.__package__)
        sys.modules[name]=module
        setattr(parent,name.rsplit(".",1)[1],module)
        exec(compile(source,path,"exec"),module.__dict__)

import numpy as np
import torch
from openpi.cache.config import CacheConfig, _dict_to_dataclass, validate_cache_config
from openpi.cache.storage_types import CachePayload
from openpi.cache.types import CheckpointID
from openpi_client import msgpack_numpy
import openpi.serving.websocket_policy_server as ws
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

configs=[]
for raw in ({}, {"enabled":True,"write_policy":{"type":"never"}}, {"key_builder":{"type":"typo"}}):
    cfg=_dict_to_dataclass(CacheConfig,raw)
    value=asdict(cfg);value.pop("adapter_binding",None)
    try:
        validate_cache_config(cfg,check_files=False)
        error=None
    except Exception as exc:
        error=str(exc)
    configs.append({"config":value,"error":error})
payload=CachePayload(torch.arange(20,dtype=torch.float32).reshape(5,4),None,None,"task",None,None)
payload.validate_for_checkpoint(CheckpointID.CP1)

class Policy:
    def __init__(self):self.lifecycle=[]
    def on_task_begin(self):self.lifecycle.append("begin")
    def on_episode_start(self,**kwargs):self.lifecycle.append("episode_start")
    def on_episode_end(self,**kwargs):self.lifecycle.append("episode_end")
    def on_task_end(self):self.lifecycle.append("end")
    def infer(self,obs):return {"actions":np.asarray(obs["state"],np.float32)[None]*2}

class Socket:
    remote_address=("127.0.0.1",1234)
    def __init__(self):
        self.packer=msgpack_numpy.Packer();self.sent=[]
        self.messages=iter(({"__ctrl__":"select_bundle","bundle_id":"default"},
             {"__ctrl__":"episode_start","__task__":"legacy"},
             {"state":np.arange(4,dtype=np.float32)}, {"__ctrl__":"episode_end"}))
    async def send(self,value):self.sent.append(value)
    async def recv(self):
        try:return self.packer.pack(next(self.messages))
        except StopIteration:raise ConnectionClosedOK(Close(1000,""),Close(1000,""),True)

policy=Policy();socket=Socket()
server=ws.WebsocketPolicyServer(policy,concurrent=True,metadata={"legacy":"metadata"},connection_policy_factory=lambda base,bid:policy)
asyncio.run(server._handler(socket))
wire=[]
for packed in socket.sent:
    value=msgpack_numpy.unpackb(packed)
    if isinstance(value,dict):value.pop("server_timing",None)
    wire.append(hashlib.sha256(msgpack_numpy.Packer().pack(value)).hexdigest())
print(json.dumps({"configs":configs,"wire_bytes_sha256":wire,"lifecycle":policy.lifecycle},sort_keys=True))
