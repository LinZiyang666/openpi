"""Offscreen worker rendering with a cross-process EGL lifecycle lock."""
from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path
import socket

STAMP = {}


@contextlib.contextmanager
def env_build_lock():
    # Share the established lock with any older worker fleet on this host.
    path = os.environ.get("RC365_ENV_BUILD_LOCK", "/tmp/rc365_env_build.lock")
    if not path:
        raise ValueError("R17 requires a cross-process environment build lock")
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def configure(renderer, device_id=0):
    os.environ.update(MUJOCO_GL=renderer, PYOPENGL_PLATFORM=renderer, CUDA_VISIBLE_DEVICES="",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="1")
    if renderer == "egl":
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
    else:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


def renderer_stamp():
    from OpenGL import GL
    values = {name: GL.glGetString(getattr(GL, name)) for name in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")}
    values = {k: v.decode() if isinstance(v, bytes) else v for k, v in values.items()}
    if not values["GL_RENDERER"]:
        raise RuntimeError("no current GL context; cannot attest worker renderer")
    if os.environ["MUJOCO_GL"] == "egl" and "NVIDIA" not in (values["GL_VENDOR"] or ""):
        raise RuntimeError(f"formal EGL workers require NVIDIA, got {values}")
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "renderer": os.environ["MUJOCO_GL"],
            "egl_device_id": int(os.environ["MUJOCO_EGL_DEVICE_ID"]) if os.environ["MUJOCO_GL"] == "egl" else None,
            **values}


class LockedEnv:
    """Serialize first render (context creation) and close; preserve env behavior."""
    def __init__(self, env):
        self._env, self._has_context = env, False

    def __getattr__(self, name):
        return getattr(self._env, name)

    def render(self):
        if not self._has_context:
            with env_build_lock():
                result = self._env.render()
                STAMP.update(renderer_stamp())
                self._has_context = True
                return result
        return self._env.render()

    def close(self):
        with env_build_lock():
            self._env.close()
        self._has_context = False


def make_env(task, idx, seed):
    from exp.metaworld import env as original
    # MT1 task sampling does not create GL contexts. Cache it before the lock,
    # so CPU-only task generation does not serialize the entire worker fleet.
    original._benchmark(task, seed)
    with env_build_lock():
        return LockedEnv(original.make_env(task, idx, seed))
