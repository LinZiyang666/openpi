"""Offscreen worker rendering with a cross-process EGL lifecycle lock."""
from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path
import socket

STAMP = {}
# Concurrent EGL lifecycle operations per host. One slot queued ~250 workers three times per episode
# (10-19: MetaWorld 15 s -> 90 s per episode on timan107); unbounded concurrency put 35 processes in
# D state there within minutes (the storm that wedged timan108). Bounded, not removed. With the shared
# EGL context (share_egl_context) a build is model compilation plus one MjrContext; 8 ran on timan1.
BUILD_SLOTS = 8


@contextlib.contextmanager
def env_build_lock():
    base = os.environ.get("RC365_ENV_BUILD_LOCK", "/tmp/rc365_env_build.lock")
    if not base:
        raise ValueError("R17 requires a cross-process environment build lock")
    paths = [Path(f"{base}.{i}") for i in range(BUILD_SLOTS)]
    handles = []
    try:
        for path in paths:
            handle = path.open("a")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                continue
        else:
            fcntl.flock(handles[os.getpid() % BUILD_SLOTS], fcntl.LOCK_EX)
        yield
    finally:
        for handle in handles:
            handle.close()


def configure(renderer, device_id=0):
    os.environ.update(MUJOCO_GL=renderer, PYOPENGL_PLATFORM=renderer, CUDA_VISIBLE_DEVICES="",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="1")
    if renderer == "egl":
        os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
        share_egl_context()
    else:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


def share_egl_context():
    """One EGL context per process and size, reused by every environment.

    gymnasium creates and destroys an EGL context per environment, i.e. per episode; that storm is
    what the build lock queues (timan108 wedged; 35 D-state processes on timan107 when unbounded).
    Frames match per-episode contexts within their own run-to-run noise (10-19: <=1 level on ~0.04%
    of pixels for both). Each environment still builds its own MjrContext inside the shared context;
    closing frees it and glFinish makes the driver return the memory (without it the pool grew to
    ~820 MiB per worker; with it 13 MiB over 150 episodes)."""
    from gymnasium.envs.mujoco import mujoco_rendering
    from mujoco.egl import GLContext
    from OpenGL import GL
    contexts = {}

    def shared(width, height):
        if (width, height) not in contexts:
            context = GLContext(width, height)
            context.free = lambda: None  # released at process exit
            contexts[(width, height)] = context
        return contexts[(width, height)]

    def free(viewer):
        con = getattr(viewer, "con", None)
        if con is not None:
            viewer.make_context_current()
            con.free()
            GL.glFinish()
    mujoco_rendering._ALL_RENDERERS["egl"] = shared
    mujoco_rendering.OffScreenViewer.free = free


def share_robosuite_egl_context():
    """LIBERO counterpart of share_egl_context: robosuite's MjRenderContext creates one EGL context per
    environment (per episode) and destroys it in __del__. Reuse one per process, size and device;
    each render context frees only its MjrContext, then glFinish returns the memory."""
    from robosuite.utils import binding_utils
    from OpenGL import GL
    native = binding_utils.GLContext
    contexts = {}

    def shared(max_width, max_height, device_id=0):
        key = (max_width, max_height, device_id)
        if key not in contexts:
            context = native(max_width=max_width, max_height=max_height, device_id=device_id)
            context.free = lambda: None  # released at process exit
            contexts[key] = context
        return contexts[key]

    def release(render_context):
        render_context.gl_ctx.make_current()
        render_context.con.free()
        GL.glFinish()
    binding_utils.GLContext = shared
    binding_utils.MjRenderContext.__del__ = release


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
