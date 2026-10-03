"""One reset and render per physical GPU, no policy requests or episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--gpus", default="0,1,2,3")
    a = ap.parse_args()
    if a.gpu is None:
        gpus = [int(g) for g in a.gpus.split(",")]
        if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
            raise ValueError("--gpus must list distinct nonnegative GPU IDs")
        for gpu in gpus:
            env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
                   "MUJOCO_EGL_DEVICE_ID": str(gpu), "CUDA_VISIBLE_DEVICES": str(gpu)}
            subprocess.run(["/scratch/zixuans8/libero_sim/bin/python", str(Path(__file__).resolve()), "--gpu", str(gpu)],
                           check=True, env=env, timeout=90)
        print("RENDER_OK GPUs=" + ",".join(map(str, gpus)))
        return
    start = time.monotonic()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    try:
        env.seed(7)
        obs = env.reset()
        assert obs["agentview_image"].shape == (256, 256, 3)
        assert obs["agentview_image"].std() > 0
        # Resolve the GL symbol from the current EGL implementation. On t107
        # PyOpenGL's GL wrapper crashes here even after a successful reset/render.
        import ctypes
        from mujoco.egl import egl_ext as EGL
        address = EGL.eglGetProcAddress(b"glGetString")
        assert address, "EGL cannot resolve glGetString"
        renderer_bytes = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_uint)(address)(0x1F01)
        assert renderer_bytes, "no current EGL render context"
        renderer = renderer_bytes.decode()
        assert "NVIDIA" in renderer or "A5000" in renderer, renderer
        print(json.dumps(dict(gpu=a.gpu, egl_device=os.environ["MUJOCO_EGL_DEVICE_ID"], renderer=renderer,
                              image_shape=list(obs["agentview_image"].shape), reset_render_s=round(time.monotonic()-start, 3))))
    finally:
        env.close()


if __name__ == "__main__":
    main()
