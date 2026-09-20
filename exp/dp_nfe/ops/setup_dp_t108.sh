#!/bin/bash
# timan108: official Diffusion Policy (real-stanford/diffusion_policy) environment via micromamba (no sudo on the box):
# conda_environment.yaml (py3.9, torch 1.12.1 cu116, free-mujoco-py 2.1.6, robosuite fork, robomimic 0.2.0) plus the
# conda-forge GL packages mujoco-py needs instead of apt (glew, mesalib, glfw, patchelf, libglu), MuJoCo 2.1.0 binaries.
# Everything under /scratch/zixuans8/dp. Idempotent.
set -u -o pipefail
export HOME=/home/zixuans8
D=/scratch/zixuans8/dp; mkdir -p $D /tmp/dp; cd $D
export MAMBA_ROOT_PREFIX=$D/mamba
if [ ! -x $D/bin/micromamba ]; then
  echo "== $(date +%H:%M:%S) micromamba"
  mkdir -p $D/bin && wget -q -O $D/micromamba.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest && tar -xjf $D/micromamba.tar.bz2 -C $D/bin --strip-components=1 bin/micromamba || { echo SETUP_FAIL micromamba; exit 1; }
  [ -x $D/micromamba ] && [ ! -x $D/bin/micromamba ] && mv $D/micromamba $D/bin/micromamba
fi
MM=$D/bin/micromamba
if [ ! -d $D/diffusion_policy/.git ]; then
  echo "== $(date +%H:%M:%S) clone"
  git clone --depth 1 https://github.com/real-stanford/diffusion_policy.git $D/diffusion_policy 2>&1 | tail -1 || { echo SETUP_FAIL clone; exit 1; }
fi
if [ ! -d $D/mujoco210 ]; then
  echo "== $(date +%H:%M:%S) mujoco210"
  wget -q -O $D/mujoco210.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xzf $D/mujoco210.tar.gz -C $D || { echo SETUP_FAIL mujoco210; exit 1; }
fi
if [ ! -x $D/env/bin/python ]; then
  echo "== $(date +%H:%M:%S) conda env from conda_environment.yaml (+ GL packages)"
  cp $D/diffusion_policy/conda_environment.yaml $D/env.yaml
  # add the GL/build packages mujoco-py needs (apt not available); keep everything else verbatim
  python3 - <<'PY'
import re
p='/scratch/zixuans8/dp/env.yaml'; s=open(p).read()
extra="  - glew\n  - mesalib\n  - glfw\n  - patchelf\n  - libglu\n  - gcc_linux-64=11\n  - gxx_linux-64=11\n"
s=s.replace("dependencies:\n","dependencies:\n"+extra,1)
open(p,'w').write(s); print("env.yaml patched")
PY
  $MM env create -y -p $D/env -f $D/env.yaml 2>&1 | tail -5 || echo "env create returned non-zero (pip part may have failed; fixed below)"
fi
# pip part, without dm-control (its mujoco binding fails to build here and robomimic tasks do not need it)
echo "== $(date +%H:%M:%S) pip packages"
$D/env/bin/pip install "ray[default,tune]==2.2.0" free-mujoco-py==2.1.6 pygame==2.1.2 pybullet-svl==3.1.6.4 \
  "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz" \
  robomimic==0.2.0 pytorchvideo==0.1.5 imagecodecs==2022.9.26 \
  "r3m @ https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz" 2>&1 | tail -2 || { echo SETUP_FAIL pip; exit 1; }
$D/env/bin/pip install -e $D/diffusion_policy 2>&1 | tail -1 || true
echo "== $(date +%H:%M:%S) mujoco-py compile check"
export MUJOCO_PY_MUJOCO_PATH=$D/mujoco210 LD_LIBRARY_PATH=$D/mujoco210/bin:$D/env/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-} CPATH=$D/env/include LIBRARY_PATH=$D/env/lib MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$D/env/bin/python -c "import mujoco_py, robosuite, robomimic, diffusers, torch; print('dp import ok torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'robosuite', robosuite.__version__)" 2>&1 | tail -3 || { echo SETUP_FAIL import; exit 1; }
echo "SETUP_DP_DONE $(date +%H:%M:%S)"
