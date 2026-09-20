#!/bin/bash
# timan107: RoboTwin 2.0 simulator env for the X-WAM RoboTwin client (follows RoboTwin's script/_install.sh; CuRobo is
# mandatory because the eval loop runs the scripted expert once per seed and every robot load builds a CuroboPlanner).
# Everything lives on the local RAID /srv/local (7.3 TB, ext4; /scratch = root fs is 97% full and the NFS home is off limits): micromamba (python 3.10 + CUDA 12.1 toolkit for nvcc) at
# $ROOT/mamba/envs/rt, X-WAM + RoboTwin submodule at $ROOT/X-WAM, assets (14 GB zipped) under the submodule. Idempotent.
# usage: setup_robotwin_t107.sh [ROOT=/srv/local/zixuans8/robotwin]
set -u -o pipefail
ROOT=${1:-/srv/local/zixuans8/robotwin}; export HOME=/home/zixuans8
export MAMBA_ROOT_PREFIX=$ROOT/mamba; MM=$ROOT/bin/micromamba; ENV=$ROOT/mamba/envs/rt
export PATH=$ENV/bin:$ROOT/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_HOME=$ENV TORCH_CUDA_ARCH_LIST="6.1" MAX_JOBS=24 PIP_CACHE_DIR=$ROOT/pip_cache HF_HOME=$ROOT/hf_home
mkdir -p $ROOT/bin $ROOT/pip_cache $HF_HOME /tmp/robotwin
say() { echo "== $(date +%H:%M:%S) $*"; }
if [ ! -x $MM ]; then
  say micromamba
  curl -Ls -o $MM https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 && chmod +x $MM || { echo SETUP_FAIL micromamba; exit 1; }
fi
if [ ! -x $ENV/bin/python ]; then
  say "env python 3.10"
  $MM create -y -p $ENV -c conda-forge python=3.10 pip 2>&1 | tail -2 || { echo SETUP_FAIL env; exit 1; }
fi
if [ ! -x $ENV/bin/nvcc ]; then
  say "cuda toolkit 12.1 (nvcc for curobo)"
  $MM install -y -p $ENV -c nvidia/label/cuda-12.1.1 cuda-toolkit 2>&1 | tail -2 || { echo SETUP_FAIL cuda; exit 1; }
fi
$ENV/bin/nvcc --version | tail -1
if [ ! -d $ROOT/X-WAM/.git ]; then
  say "clone X-WAM"
  git clone https://github.com/sharinka0715/X-WAM.git $ROOT/X-WAM 2>&1 | tail -1 || { echo SETUP_FAIL clone; exit 1; }
fi
cd $ROOT/X-WAM
if [ ! -f third_party/RoboTwin/script/_install.sh ]; then
  say "submodule RoboTwin"
  git submodule update --init third_party/RoboTwin 2>&1 | tail -1 || { echo SETUP_FAIL submodule; exit 1; }
fi
RT=$ROOT/X-WAM/third_party/RoboTwin
PY=$ENV/bin/python
if ! $PY -c "import torch; assert torch.__version__.startswith('2.4.1')" 2>/dev/null; then
  say "torch 2.4.1 cu121 (sm_61 for the GTX 1080s)"
  $PY -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1 || { echo SETUP_FAIL torch; exit 1; }
fi
if ! $PY -c "import sapien, mplib, toppra, zmq, tyro" 2>/dev/null; then
  say "RoboTwin requirements (minus torch/hf pins) + client deps"
  grep -vE "^(torch|torchvision|huggingface_hub|azure|wandb|openai|ffmpeg)" $RT/script/requirements.txt > /tmp/robotwin/req.txt
  $PY -m pip install -r /tmp/robotwin/req.txt 2>&1 | tail -1 || { echo SETUP_FAIL requirements; exit 1; }
  $PY -m pip install toppra pyzmq tyro imageio[ffmpeg] "numpy==1.23.5" huggingface_hub 2>&1 | tail -1 || { echo SETUP_FAIL deps; exit 1; }
fi
# _install.sh's two source patches (sapien urdf loader utf-8 + srdf suffix; mplib screw-plan collision check)
SAP=$($PY -c "import sapien, os; print(os.path.dirname(sapien.__file__))")/wrapper/urdf_loader.py
grep -q 'encoding="utf-8"' $SAP || sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAP
grep -q '\[:-4\] + ".srdf"' $SAP || sed -i 's/urdf_file\[:-4\] + "srdf"/urdf_file[:-4] + ".srdf"/' $SAP
MPL=$($PY -c "import mplib, os; print(os.path.dirname(mplib.__file__))")/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPL
if ! $PY -c "import curobo" 2>/dev/null; then
  say "curobo v0.7.8 (build, arch 6.1)"
  cd $RT/envs
  [ -d curobo/.git ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git 2>&1 | tail -1
  cd curobo && $PY -m pip install -e . --no-build-isolation > /tmp/robotwin/curobo_build.log 2>&1 || { tail -30 /tmp/robotwin/curobo_build.log; echo SETUP_FAIL curobo; exit 1; }
  $PY -m pip install warp-lang==1.12.0 setuptools==69.5.1 2>&1 | tail -1
  cd $ROOT/X-WAM
fi
$PY -m pip install "numpy==1.23.5" 2>&1 | tail -1
# GTX 1080 (sm_61): OIDN's CUDA device needs Volta+ ("unsupported device type: CUDA", frames left un-denoised); the
# driver's OptiX denoiser works on Pascal (probed 37 ms/frame at 32 spp 320x240) -> use it for every scene
sed -i 's/set_ray_tracing_denoiser("oidn")/set_ray_tracing_denoiser("optix")/' $RT/envs/_base_task.py $RT/script/test_render.py
grep -c 'denoiser("optix")' $RT/envs/_base_task.py
$PY -c "import sapien, mplib, curobo, torch, warp, numpy; print('import ok sapien', sapien.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)" || { echo SETUP_FAIL import; exit 1; }
if [ ! -d $RT/assets/objects ] || [ ! -d $RT/assets/embodiments ] || [ ! -d $RT/assets/background_texture ]; then
  say "assets (background_texture 10 GB + objects 3.5 GB + embodiments)"
  cd $RT/assets && $PY _download.py 2>&1 | tail -2 || { echo SETUP_FAIL assets_dl; exit 1; }
  for z in background_texture embodiments objects; do [ -f $z.zip ] && { unzip -q -o $z.zip && rm -f $z.zip; }; done
  cd $RT && $PY ./script/update_embodiment_config_path.py 2>&1 | tail -1
  cd $ROOT/X-WAM
fi
ls $RT/assets
say "render smoke (Sapien_TEST headless)"
cd $RT && $PY -c "import sys; sys.path.insert(0,'script'); from test_render import Sapien_TEST; Sapien_TEST(); print('SAPIEN_RENDER_OK')" 2>&1 | tail -3
echo "SETUP_ROBOTWIN_DONE $(date +%H:%M:%S)"
