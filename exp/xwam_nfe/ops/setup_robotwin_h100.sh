#!/bin/bash
# h100: RoboTwin 2.0 simulator env for the X-WAM RoboTwin client (owner 12:5x: h100 also hosts clients next to one policy
# server; H100 exposes Vulkan RT to SAPIEN, probed 9 ms/frame at 32 spp). Same recipe as setup_robotwin_wls.sh; the root fs
# has 7 GB left so everything (micromamba, env, code, assets, caches, results) lives under /data: $ROOT = code + env,
# $DATA = assets (symlinked into the submodule) + caches + results. sm_90, CUDA 12.4 toolkit from micromamba, torch 2.5.1 cu124.
# usage: setup_robotwin_h100.sh [ROOT=/data/xwam/robotwin] [DATA=/data/xwam/robotwin_data]
set -u -o pipefail
ROOT=${1:-/data/xwam/robotwin}; DATA=${2:-/data/xwam/robotwin_data}; export HOME=/home/exouser
export MAMBA_ROOT_PREFIX=$ROOT/mamba; MM=$ROOT/bin/micromamba; ENV=$ROOT/mamba/envs/rt
export PATH=$ENV/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_HOME=$ENV TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 PIP_CACHE_DIR=$DATA/cache/pip HF_HOME=$DATA/cache/hf
mkdir -p $ROOT/bin $DATA/cache/pip $DATA/cache/hf $DATA/assets $DATA/results /tmp/robotwin
export XDG_CACHE_HOME=$DATA/cache/xdg   # warp / nvidia caches off the 7 GB root fs
say() { echo "== $(date +%H:%M:%S) $*"; }
[ -x $MM ] || { say micromamba; curl -Ls -o $MM https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 && chmod +x $MM || { echo SETUP_FAIL micromamba; exit 1; }; }
if [ ! -x $ENV/bin/python ]; then
  say "env python 3.10"
  $MM create -y -p $ENV -c conda-forge python=3.10 pip 2>&1 | tail -2 || { echo SETUP_FAIL env; exit 1; }
fi
if [ ! -x $ENV/bin/nvcc ]; then
  say "cuda toolkit 12.4 (nvcc for curobo; system nvcc is 13.2 and would mismatch torch)"
  $MM install -y -p $ENV -c nvidia/label/cuda-12.4.1 cuda-toolkit 2>&1 | tail -2 || { echo SETUP_FAIL cuda; exit 1; }
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
if ! $PY -c "import torch; assert torch.__version__.startswith('2.5.1')" 2>/dev/null; then
  say "torch 2.5.1 cu124"
  $PY -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1 || { echo SETUP_FAIL torch; exit 1; }
fi
if ! $PY -c "import sapien, mplib, toppra, zmq, tyro" 2>/dev/null; then
  say "RoboTwin requirements (minus torch/hf pins) + client deps"
  grep -vE "^(torch|torchvision|huggingface_hub|azure|wandb|openai|ffmpeg)" $RT/script/requirements.txt > /tmp/robotwin/req.txt
  $PY -m pip install -r /tmp/robotwin/req.txt 2>&1 | tail -1 || { echo SETUP_FAIL requirements; exit 1; }
  $PY -m pip install toppra pyzmq tyro "imageio[ffmpeg]" "numpy==1.23.5" huggingface_hub "setuptools<70" 2>&1 | tail -1 || { echo SETUP_FAIL deps; exit 1; }
fi
# _install.sh's two source patches (sapien urdf loader utf-8 + srdf suffix; mplib screw-plan collision check)
SAP=$($PY -c "import sapien, os; print(os.path.dirname(sapien.__file__))")/wrapper/urdf_loader.py
grep -q 'encoding="utf-8"' $SAP || sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $SAP
grep -q '\[:-4\] + ".srdf"' $SAP || sed -i 's/urdf_file\[:-4\] + "srdf"/urdf_file[:-4] + ".srdf"/' $SAP
MPL=$($PY -c "import mplib, os; print(os.path.dirname(mplib.__file__))")/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $MPL
if ! $PY -c "import curobo" 2>/dev/null; then
  say "curobo v0.7.8 (build, arch 9.0)"
  cd $RT/envs
  [ -d curobo/.git ] || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git 2>&1 | tail -1
  cd curobo && $PY -m pip install -e . --no-build-isolation > /tmp/robotwin/curobo_build.log 2>&1 || { tail -30 /tmp/robotwin/curobo_build.log; echo SETUP_FAIL curobo; exit 1; }
  $PY -m pip install warp-lang==1.12.0 setuptools==69.5.1 2>&1 | tail -1
  cd $ROOT/X-WAM
fi
$PY -m pip install "numpy==1.23.5" 2>&1 | tail -1
$PY -c "import sapien, mplib, curobo, torch, warp, numpy; print('import ok sapien', sapien.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'numpy', numpy.__version__)" || { echo SETUP_FAIL import; exit 1; }
# assets live on /data; the submodule's assets dir (holds _download.py + files/) becomes a symlink to them
if [ ! -L $RT/assets ]; then
  say "move assets dir to $DATA/assets (symlink)"
  cp -a $RT/assets/. $DATA/assets/ && rm -r $RT/assets </dev/null && ln -s $DATA/assets $RT/assets
fi
if [ ! -d $DATA/assets/objects ] || [ ! -d $DATA/assets/embodiments ] || [ ! -d $DATA/assets/background_texture ]; then
  say "assets (background_texture 10 GB + objects 3.5 GB + embodiments) -> $DATA/assets"
  cd $DATA/assets && $PY _download.py 2>&1 | tail -2 || { echo SETUP_FAIL assets_dl; exit 1; }
  for z in background_texture embodiments objects; do [ -f $z.zip ] && { unzip -q -o $z.zip && rm $z.zip; }; done
  cd $RT && $PY ./script/update_embodiment_config_path.py 2>&1 | tail -1
  cd $ROOT/X-WAM
fi
ls $RT/assets/
say "render smoke (Sapien_TEST headless)"
cd $RT && $PY -c "import sys; sys.path.insert(0,'script'); from test_render import Sapien_TEST; Sapien_TEST(); print('SAPIEN_RENDER_OK')" 2>&1 | tail -3
echo "SETUP_ROBOTWIN_DONE $(date +%H:%M:%S)"
