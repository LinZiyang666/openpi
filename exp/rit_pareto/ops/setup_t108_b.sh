#!/bin/bash
# setup_t108_b.sh — Phase B: LIBERO conda prefix from weilandserver, relocated to /scratch/zixuans8/libero_sim.
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
S=http://ziyanglin.com:23161
P=/scratch/zixuans8/libero_sim
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
step download
curl -sS -m 1800 -o /tmp/libero_sim.tgz $S/libero_sim.tgz && curl -sS -m 60 -o /tmp/libero_sim.tgz.sha $S/libero_sim.tgz.sha || { echo SETUP_FAILED download; exit 1; }
cd /tmp && sed "s#/data/rit_stage/##" libero_sim.tgz.sha | sha256sum -c - || { echo SETUP_FAILED sha; exit 1; }
step extract
rm -rf $P && mkdir -p /scratch/zixuans8 && tar -xzf /tmp/libero_sim.tgz -C /scratch/zixuans8 || { echo SETUP_FAILED extract; exit 1; }
step relocate
# EGL: timan108 has the 535 user-space vendor staged at /scratch/zixuans8/nvidia-gl (json + lib) and a system 10_nvidia.json;
# replace weilandserver's 595 hook with the timan108 vendor dir.
cat > $P/etc/conda/activate.d/nvidia_egl.sh <<'HOOK'
# timan108: NVIDIA EGL user-space vendor matching driver 535.183.01 (staged under /scratch/zixuans8/nvidia-gl).
export __EGL_VENDOR_LIBRARY_DIRS=/scratch/zixuans8/nvidia-gl
export LD_LIBRARY_PATH=/scratch/zixuans8/nvidia-gl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
HOOK
# openpi_client editable install -> point at the dispatch clone's package source.
cd $P/lib/python3.8/site-packages
for f in __editable__*openpi_client*.pth easy-install.pth openpi_client.egg-link __editable___openpi_client*finder.py; do
  [ -f "$f" ] && { echo "editable file: $f"; sed -i "s#/home/weiland/openpi/packages/openpi-client#/scratch/zixuans8/openpi_dispatch/packages/openpi-client#g" "$f"; }
done
grep -rl "/home/weiland" *.pth *.py 2>/dev/null | head -5
step smoke-import
/scratch/zixuans8/dsp_bin/conda run --no-capture-output -p $P python -c "import libero, mujoco, robosuite, openpi_client; print('libero import ok', mujoco.__version__)" || { echo SETUP_FAILED import; exit 1; }
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 /scratch/zixuans8/dsp_bin/conda run --no-capture-output -p $P python -c "
import os; os.environ['MUJOCO_GL']='egl'
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\".1 .1 .1\"/></worldbody></mujoco>')
d = mujoco.MjData(m); r = mujoco.Renderer(m, 64, 64); r.update_scene(d); img = r.render(); print('egl render ok', img.shape, int(img.max()))
" || { echo SETUP_FAILED egl; exit 1; }
echo SETUP_B_DONE
