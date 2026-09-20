#!/bin/bash
# h100 (root fs ~7 GB free, 1 TB volume at /data, NVIDIA EGL present, free sudo not used): replicate the weilandserver
# Diffusion Policy environment (setup_dp_wls.sh recipe: conda_environment.yaml py3.9 / torch 1.12.1 cu116 / free-mujoco-py
# 2.1.6 / robosuite fork / robomimic 0.2.0 + conda-forge GL packages, MuJoCo 2.1.0, pip extras without dm-control) entirely
# under /data/dp_h100 with every cache redirected off the root fs. Idempotent. usage: setup_dp_h100.sh <dp revision>
set -u -o pipefail
REV=${1:?dp revision}
export HOME=/home/exouser
B=/data/dp_h100; mkdir -p $B/data $B/tmp $B/pipcache $B/xdg /tmp/x0; cd $B
export MAMBA_ROOT_PREFIX=$B/mamba PIP_CACHE_DIR=$B/pipcache XDG_CACHE_HOME=$B/xdg TMPDIR=$B/tmp
say() { echo "== $(date +%H:%M:%S) $*"; }
MM=$B/bin/micromamba
if [ ! -x $MM ]; then
  mkdir -p $B/bin
  if [ -x /data/xwam/robotwin/bin/micromamba ]; then cp /data/xwam/robotwin/bin/micromamba $MM; else
    say micromamba; wget -q -O $B/micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest && tar -xjf $B/micromamba.tar.bz2 -C $B/bin --strip-components=1 bin/micromamba || { echo SETUP_FAIL micromamba; exit 1; }
  fi
fi
if [ ! -d $B/diffusion_policy/.git ]; then
  say clone; git clone https://github.com/real-stanford/diffusion_policy.git $B/diffusion_policy 2>&1 | tail -1 || { echo SETUP_FAIL clone; exit 1; }
fi
git -C $B/diffusion_policy checkout -q $REV && say "dp rev $(git -C $B/diffusion_policy rev-parse HEAD)"
if [ ! -d $B/mujoco210 ]; then
  say mujoco210; wget -q -O $B/mujoco210.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xzf $B/mujoco210.tar.gz -C $B || { echo SETUP_FAIL mujoco210; exit 1; }
fi
if [ ! -x $B/env/bin/python ]; then
  say "conda env from conda_environment.yaml (+ GL packages)"
  cp $B/diffusion_policy/conda_environment.yaml $B/env.yaml
  python3 - <<'PY'
p='/data/dp_h100/env.yaml'; s=open(p).read()
extra="  - glew\n  - mesalib\n  - glfw\n  - patchelf\n  - libglu\n  - gcc_linux-64=11\n  - gxx_linux-64=11\n"
s=s.replace("dependencies:\n","dependencies:\n"+extra,1)
open(p,'w').write(s); print("env.yaml patched")
PY
  $MM env create -y -p $B/env -f $B/env.yaml 2>&1 | tail -5 || echo "env create returned non-zero (pip part may have failed; fixed below)"
fi
say "pip packages"
$B/env/bin/pip install "ray[default,tune]==2.2.0" free-mujoco-py==2.1.6 pygame==2.1.2 pybullet-svl==3.1.6.4 \
  "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz" \
  robomimic==0.2.0 pytorchvideo==0.1.5 imagecodecs==2022.9.26 \
  "r3m @ https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz" 2>&1 | tail -2 || { echo SETUP_FAIL pip; exit 1; }
$B/env/bin/pip install -e $B/diffusion_policy 2>&1 | tail -1 || true
[ -e $B/env/lib/libGL.so ] || ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 $B/env/lib/libGL.so
[ -e $B/env/lib/libEGL.so ] || ln -s /usr/lib/x86_64-linux-gnu/libEGL.so.1 $B/env/lib/libEGL.so
$B/env/bin/pip install "huggingface_hub==0.20.3" 2>&1 | tail -1
# kitchen env (adept_envs) needs dm_control; the mujoco wheel must be pinned (newer ones try to build from source on py3.9)
$B/env/bin/pip install "mujoco==2.3.7" 2>&1 | tail -1; $B/env/bin/pip install "dm-control==1.0.9" 2>&1 | tail -1
say "mujoco-py compile check"
export PATH=$B/env/bin:$PATH MUJOCO_PY_MUJOCO_PATH=$B/mujoco210 LD_LIBRARY_PATH=$B/mujoco210/bin:$B/env/lib:${LD_LIBRARY_PATH:-} CPATH=$B/env/include LIBRARY_PATH=$B/env/lib MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$B/env/bin/python -c "import mujoco_py, robosuite, robomimic, diffusers, torch; print('dp import ok torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'robosuite', robosuite.__version__)" 2>&1 | tail -3 || { echo SETUP_FAIL import; exit 1; }
du -sh $B | cut -f1 | xargs echo "dp_h100 size"; df -h / | tail -1
echo "SETUP_DP_DONE $(date +%H:%M:%S)"
