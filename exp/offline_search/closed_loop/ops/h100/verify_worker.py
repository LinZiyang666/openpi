"""Worker import and frozen official-init attestation; no policy request."""
import json
import os
from pathlib import Path
import subprocess

R = Path(__file__).resolve().parent.parent


def main():
    env = {**os.environ, "HOME": "/home/zixuans8", "LIBERO_CONFIG_PATH": str(R / "os_cl/libero"),
           "PYTHONPATH": f"{R}:{R}/src:{R}/packages/openpi-client/src", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "egl"}
    simcode = "import sys,openpi_client,examples.libero.main; print(sys.version); print(openpi_client.__file__); print(examples.libero.main.__file__)"
    subprocess.run(["/scratch/zixuans8/libero_sim/bin/python", "-c", simcode], env=env, check=True)
    code = '''
import json
from pathlib import Path
from exp.ablation_study.cache_size.run_size_eval import load_apool_digest
for suite in ("libero_spatial", "libero_10"):
    p = Path.cwd() / "exp/ablation_study/cache_size/config" / ("apool_" + suite + ".yaml")
    value = load_apool_digest(str(p), required=True, verify_contents=True)
    expected = {"libero_spatial": "0eeece46a08b958efe7b7db4e6b13d3269b0433be4e20fbae3c0f352bc3aca9c",
                "libero_10": "52457a37eb26f9511b708f2e2efb2c175d0a1f8665ba8d57180556693c1ee756"}
    assert value["total_inits"] == 500, value
    assert value["rollup_sha256"] == expected[suite], value
    print("APOOL_OK " + suite + " total=" + str(value["total_inits"]) + " sha=" + value["rollup_sha256"])
from exp.offline_search.closed_loop.devset import pool_record
for suite in ("libero_spatial", "libero_10"):
    p, frozen = pool_record(suite, 'B')
    value = load_apool_digest(str(p), required=True, verify_contents=True)
    assert value['init_pool'] == 'B' and value['total_inits'] == 500
    print('BPOOL_OK ' + suite + ' total=500 sha=' + value['rollup_sha256'])
'''
    subprocess.run(["/scratch/zixuans8/openpi/.venv/bin/python", "-c", code], env=env, check=True, cwd=R)
    print("WORKER_IMPORTS_OK")


if __name__ == "__main__":
    main()
