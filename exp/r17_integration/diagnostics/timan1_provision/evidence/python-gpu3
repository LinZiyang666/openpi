#!/usr/bin/python3
"""Exec the unchanged venv inside a private namespace with GPUs 0-2 masked."""
import os
import pathlib
import sys

ROOT=pathlib.Path('/scratch/zixuans8/mw_r17/timan1_provision')
UUID='GPU-ba276422-a0b1-e371-8327-f744be942f15'
# NVML GPU3 / PCI C1:00.0 maps to NVIDIA device minor 0 on this node.
GPU_MINOR=0
info=pathlib.Path('/proc/driver/nvidia/gpus/0000:c1:00.0/information').read_text()
if UUID not in info or 'Device Minor:' not in info or int(info.split('Device Minor:',1)[1].split()[0]) != GPU_MINOR:
    raise SystemExit('GPU3 UUID/PCI/device-minor mapping changed; refusing')
if os.uname().nodename.split('.')[0]!='timan1': raise SystemExit('GPU3 wrapper requires timan1')
env=dict(os.environ)
env.setdefault('MUJOCO_EGL_DEVICE_ID','3'); env.setdefault('CUDA_VISIBLE_DEVICES',UUID)
env.setdefault('MUJOCO_GL','egl'); env.setdefault('PYOPENGL_PLATFORM','egl')
env['PYTHONPATH']=str(ROOT/'gpu3_guard')+':'+env.get('PYTHONPATH','')
env['R17_GPU3_DEVICE_ISOLATION']='1'
os.environ.update(env)
# Check before starting any application or loading site-packages.
path=ROOT/'gpu3_guard/sitecustomize.py'
exec(compile(path.read_text(),str(path),'exec'),{'__name__':'gpu3_preflight','__file__':str(path)})
argv=['/usr/bin/bwrap','--unshare-user','--die-with-parent','--bind','/','/','--dev-bind','/dev','/dev']
for dev in pathlib.Path('/dev').glob('nvidia[0-9]*'):
    if dev.name!='nvidia'+str(GPU_MINOR): argv+=['--ro-bind','/dev/null',str(dev)]
pci=pathlib.Path('/sys/bus/pci/devices/0000:c1:00.0/drm')
allowed={p.name for p in pci.iterdir()} if pci.exists() else set()
for dev in pathlib.Path('/dev/dri').glob('*'):
    if dev.is_char_device() and dev.name not in allowed: argv+=['--ro-bind','/dev/null',str(dev)]
argv+=['--','/scratch/zixuans8/openpi/.venv/bin/python',*sys.argv[1:]]
os.execve(argv[0],argv,env)
