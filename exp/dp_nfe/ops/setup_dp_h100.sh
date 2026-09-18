#!/bin/bash
# h100: replicate the weilandserver DP environment under /data/dp_h100 (root fs has ~7 GB): micromamba env from the official
# conda_environment.yaml with the same substitutions as setup_dp_wls.sh, mujoco210, the DP repo at the same revision.
# usage: setup_dp_h100.sh <dp revision>
set -u -o pipefail
REV=${1:?dp revision}
B=/data/dp_h100; MM=/data/xwam/robotwin/bin/micromamba; export MAMBA_ROOT_PREFIX=$B/mamba HOME=/home/exouser
mkdir -p $B/data /tmp/x0
say() { echo "== $(date +%H:%M:%S) $*"; }
[ -d $B/diffusion_policy/.git ] || { say clone; git clone https://github.com/real-stanford/diffusion_policy.git $B/diffusion_policy 2>&1 | tail -1; }
git -C $B/diffusion_policy checkout -q $REV && say "dp rev $(git -C $B/diffusion_policy rev-parse HEAD)"
if [ ! -x $B/env/bin/python ]; then
  say "env from conda_environment.yaml (+ mesa/glew/patchelf substitutes)"
  sed -e 's/^  - cudatoolkit.*/  - cudatoolkit=11.6/' $B/diffusion_policy/conda_environment.yaml > /tmp/x0/env.yaml
  $MM create -y -p $B/env -f /tmp/x0/env.yaml 2>&1 | tail -2 || { echo SETUP_FAIL env; exit 1; }
  $MM install -y -p $B/env -c conda-forge glew mesalib glfw patchelf libglu gcc=11 gxx=11 2>&1 | tail -1
fi
if [ ! -d $B/mujoco210 ]; then
  say mujoco210; wget -q -O /tmp/x0/mj.tgz https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && tar xzf /tmp/x0/mj.tgz -C $B
fi
export PATH=$B/env/bin:$PATH MUJOCO_PY_MUJOCO_PATH=$B/mujoco210 LD_LIBRARY_PATH=$B/mujoco210/bin:$B/env/lib
ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 $B/env/lib/libGL.so 2>/dev/null; ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1 $B/env/lib/libEGL.so 2>/dev/null
$B/env/bin/pip install -q "huggingface_hub==0.20.3" 2>&1 | tail -1
$B/env/bin/python -c "import diffusion_policy, diffusers, torch, mujoco_py; print('import ok diffusers', diffusers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo SETUP_FAIL import; exit 1; }
echo "SETUP_DP_H100_DONE $(date +%H:%M:%S)"
