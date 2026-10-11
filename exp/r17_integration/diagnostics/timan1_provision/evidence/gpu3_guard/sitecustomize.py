"""Owner-scoped timan1 guard: reject disallowed devices before simulator imports."""
import json
import os
import sys

UUID = 'GPU-ba276422-a0b1-e371-8327-f744be942f15'

def deny(message):
    os.write(2, ('TIMAN1_GPU3_GUARD: '+message+'; refusing before renderer import\n').encode())
    os._exit(78)

def check():
    if os.uname().nodename.split('.')[0] != 'timan1':
        deny('this guard is only for timan1')
    if os.environ.get('MUJOCO_EGL_DEVICE_ID') != '3':
        deny('physical EGL device must be 3')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', UUID):
        deny('CUDA visibility must be disabled or pinned to physical GPU 3')
    if os.environ.get('MUJOCO_GL') != 'egl' or os.environ.get('PYOPENGL_PLATFORM') != 'egl':
        deny('EGL backend is required')
    argv = sys.orig_argv if hasattr(sys, 'orig_argv') else sys.argv
    for i, arg in enumerate(argv):
        if arg == '--egl-device' and (i+1 == len(argv) or argv[i+1] != '3'):
            deny('CLI physical EGL device must be 3')
        if arg.startswith('--egl-device=') and arg != '--egl-device=3':
            deny('CLI physical EGL device must be 3')
    if 'exp.r17_integration.ops.formal_workers' in argv:
        try:
            path = argv[argv.index('--campaign')+1]
            with open(path) as f: campaign = json.load(f)
            if campaign.get('worker_node') != 'timan1' or campaign.get('worker_egl_device') != 3:
                deny('campaign must name timan1 and physical EGL 3')
            if campaign.get('worker_egl_devices') != [3]:
                deny('campaign EGL allowlist must be exactly [3]')
            if any(a.get('device') != 3 for a in campaign.get('worker_gpu_assignments', [])):
                deny('campaign assignments contain a disallowed physical GPU')
        except (OSError, ValueError, IndexError, KeyError) as exc:
            deny('invalid campaign: '+str(exc))

check()
# Recheck if application code changes the environment before importing native EGL.
class EGLAliasLoader:
    def __init__(self, original): self.original = original
    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, 'create_module') else None
    def exec_module(self, module):
        self.original.exec_module(module)
        original_query = module.eglQueryDevicesEXT
        def physical_gpu3_devices(max_devices=10):
            check()
            import ctypes
            query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
                module.eglGetProcAddress('eglQueryDeviceStringEXT'))
            devices = original_query(max_devices=max(max_devices, 16))
            native = [(i, d) for i, d in enumerate(devices)
                      if b'EGL_NV_device_cuda' in (query_string(d, 0x3055) or b'')]
            if len(native) != 1:
                deny('isolated EGL must expose exactly one NVIDIA device')
            query_binary = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))(module.eglGetProcAddress('eglQueryDeviceBinaryEXT'))
            value = (ctypes.c_ubyte * 16)(); size = ctypes.c_int()
            if not query_binary(native[0][1], 0x335C, 16, value, ctypes.byref(size)) or size.value != 16:
                deny('EGL physical device UUID query failed')
            actual = bytes(value).hex()
            if actual != UUID.removeprefix('GPU-').replace('-', ''):
                deny('isolated EGL UUID differs from physical GPU3: '+actual)
            os.environ['R17_GPU3_EGL_UUID_HEX'] = actual
            os.environ['R17_GPU3_EGL_NATIVE_INDEX'] = str(native[0][0])
            # Masked devices disappear from EGL discovery. Preserve the formal
            # physical ordinal 3 by aliasing only that sole NVIDIA device.
            return [native[0][1]] * 4
        module.eglQueryDevicesEXT = physical_gpu3_devices

class RendererGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('mujoco', 'OpenGL', 'robosuite', 'libero') or fullname.endswith('egl_context'):
            check()
            if os.environ.get('R17_GPU3_DEVICE_ISOLATION') != '1':
                deny('native imports require the GPU3 device namespace wrapper')
            null_dev = os.stat('/dev/null').st_rdev
            for ordinal in (1, 2, 3):
                if os.stat('/dev/nvidia'+str(ordinal)).st_rdev != null_dev:
                    deny('protected device '+str(ordinal)+' is not masked')
            if os.stat('/dev/nvidia0').st_rdev != os.makedev(195, 0):
                deny('physical GPU3 device minor 0 is unavailable')
        if fullname == 'mujoco.egl.egl_ext':
            from importlib.machinery import PathFinder
            spec = PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                deny('MuJoCo EGL extension loader missing')
            spec.loader = EGLAliasLoader(spec.loader)
            return spec
        return None
sys.meta_path.insert(0, RendererGuard())

# NVML admission stays in the host namespace and queries only physical GPU3.
# Every native worker child enters the device namespace before Python startup.
if os.environ.get('R17_GPU3_DEVICE_ISOLATION') != '1':
    import subprocess
    class GPU3Popen(subprocess.Popen):
        def __init__(self, args, *positional, **kwargs):
            argv = list(args) if isinstance(args, (list, tuple)) else None
            modules = ('exp.r17_integration.conductor.worker_entry',
                       'exp.r17_integration.ops.worker_lifetime',
                       'exp.r17_integration.ops.cache_study_worker',
                       'examples.libero.worker_entry')
            native_worker = argv and (any(m in argv for m in modules) or
                '--child' in argv and any('r17_t1_render_probe' in str(a) for a in argv))
            if native_worker:
                argv[0] = '/scratch/zixuans8/mw_r17/timan1_provision/python-gpu3'
                args = argv
            super().__init__(args, *positional, **kwargs)
    subprocess.Popen = GPU3Popen
