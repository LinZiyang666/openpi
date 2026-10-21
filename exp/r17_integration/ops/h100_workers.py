"""LIBERO workers on the model-server machine itself (node 'h100').

Owner 2026-10-18: workers started on a machine serve that machine's own server
first. Node 'h100' is whatever machine the lane calls 'h100' (the H100, or its
R17_H100_NODE stand-in), so its workers reach the server over loopback.

The environment is the local node's, unpacked under ROOT: the same Python
3.11.15 build, the same base packages (numpy, mujoco, torch, ...), the same
LIBERO dependency release and asset release. `setup` installs it once per
machine; nothing outside ROOT is touched.
"""
import json
from pathlib import Path

ROOT = '/data/mw_finetune/qualification/h100_workers'
ENV_TAR = ROOT+'/env.tar'
PYTHON = ROOT+'/env311/bin/python'
HOME = ROOT+'/home'
DEPENDENCY = ROOT+'/dependencies/libero311_42f0e80603054ed9f4bab84a6ce71bdbc71958c6c5c0329f9b5c892d917b3fe9'
ASSETS = ROOT+'/assets/libero_3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc/assets'
CONFIG_SOURCE = ROOT+'/original_paths.yaml'
# One renderer per core: the H100 machines' 20 cores are shared with the model
# server; the A100 (owner 10-18: worker machine only) has 32.
CPU_WORKERS = {'h100': 40, 'a100': 80}  # workers mostly wait on inference (A100: 29 workers, load 5)
NODES = tuple(CPU_WORKERS)

SETUP = r'''
set -euo pipefail
R=%(root)s
cd $R
if [ ! -f $R/.unpacked ]; then tar -xf %(tar)s -C $R && touch $R/.unpacked; fi
[ -x $R/env311/bin/python ] || $R/python311/bin/python3.11 -m venv --without-pip $R/env311
site=$($R/env311/bin/python -c 'import site; print(site.getsitepackages()[0])')
echo $R/base_site > $site/r17_readonly_base.pth
mkdir -p %(home)s/.cache
[ -d $R/extra ] || true
for x in $R/extra*.tar; do [ -f "$x" ] && [ ! -f "$x.unpacked" ] && tar -xf "$x" -C $R/base_site && touch "$x.unpacked"; done
# The release links its assets by the local node's absolute path; point it here.
chmod u+w %(dependency)s/libero/libero
ln -sfn %(assets)s %(dependency)s/libero/libero/assets
ln -sfn %(assets)s %(home)s/.cache/libero_assets
cat > %(config)s <<YAML
assets: %(assets)s
bddl_files: %(dependency)s/libero/libero/bddl_files
benchmark_root: %(dependency)s/libero/libero
datasets: %(dependency)s/libero/libero/../datasets
init_states: %(dependency)s/libero/libero/init_files
YAML
$R/env311/bin/python -c 'import numpy, mujoco, torch, robosuite' 2>/dev/null || PYTHONPATH=%(dependency)s $R/env311/bin/python -c 'import numpy, mujoco, torch, robosuite; print("imports ok")'
'''


def setup_script():
    return SETUP % dict(root=ROOT, tar=ENV_TAR, home=HOME, assets=ASSETS, config=CONFIG_SOURCE,
                        dependency=DEPENDENCY)


def install(remote, payload, node='h100'):
    """The unpacked dependency/asset releases, as install_worker_dependencies returns them."""
    remote(node, ['bash', '-c', setup_script()], 900)
    return dict(private_dependencies=DEPENDENCY, assets=ASSETS, config_source=CONFIG_SOURCE,
                copied=['libero', 'robosuite', 'bddl', 'easydict', 'robomimic', 'thop', 'future', 'past'],
                content_identity=DEPENDENCY.rsplit('_', 1)[1], asset_release=ASSETS.rsplit('/', 1)[0])


def route(worker, campaign):
    node = worker.get('worker_node')
    if node not in NODES:
        return worker
    from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import SIZES
    vram, ram = SIZES[campaign['domain']]
    worker.update(worker_egl_device=0, worker_egl_devices=[0], worker_vram_mib=vram,
                  worker_memory_mib=ram, worker_ram_reservation_mib=ram,
                  # On the server's machine the card also runs the model: keep a little room for its growth.
                  # h100-temp gets no own-machine renderers (Xid 31, defaults.apply). Owner 10-20: aggressive,
                  # a few GB left is enough (1 GiB here + the 2 GiB card reserve); 4 GiB left GR00T R11 none.
                  worker_server_vram_headroom_mib=1024 if node == 'h100' else 0,
                  policy_endpoint=('127.0.0.1:'+str(campaign['public_port']) if node == 'h100'
                                   else campaign['endpoint']))
    return worker
