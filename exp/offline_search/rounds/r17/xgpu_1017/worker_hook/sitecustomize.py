"""Only NEW workers with this directory on PYTHONPATH and XGPU_TRACE_ROOT opt in."""
import os
if os.environ.get('XGPU_TRACE_ROOT'):
    try:
        from client_trace import from_environment
        from_environment()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)  # a requested evidence hook must not fail silently
