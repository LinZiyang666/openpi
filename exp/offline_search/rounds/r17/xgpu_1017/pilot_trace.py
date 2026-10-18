"""By-value instrumentation of native fleet children, including recycled workers.

Retains the original Python executable, GPU2 namespace wrapper, worker module,
argv and environment. No replacement worker archive or sitecustomize is used.
"""
def child_command(argv, trace_source):
    argv = list(argv)
    # The actual child emitted by formal_workers always uses -m or -c.
    pos = next((i for i, x in enumerate(argv) if x in ('-m', '-c')), None)
    if pos is None:raise ValueError('unrecognised native worker command')
    prefix = ('import os,sys,types,runpy\n'
              'm=types.ModuleType("xgpu_client_trace");'
              'exec(compile('+repr(trace_source)+',"xgpu_client_trace","exec"),m.__dict__);'
              'm.from_environment()\n')
    if argv[pos] == '-m':
        body = 'sys.argv[0]='+repr(argv[pos+1])+';runpy.run_module('+repr(argv[pos+1])+',run_name="__main__",alter_sys=True)'
    else:body = 'exec(compile('+repr(argv[pos+1])+',"native_worker_command","exec"),{"__name__":"__main__"})'
    return argv[:pos]+['-c',prefix+body]+argv[pos+2:]


def fleet_prefix(trace_source):
    import inspect
    return (inspect.getsource(child_command)+'\n'
        'from exp.r17_integration.ops import formal_workers as _xgpu_fleet\n'
        '_xgpu_native_main=_xgpu_fleet.main\n'
        'def _xgpu_main():\n'
        '    command=_xgpu_fleet.worker_command\n'
        '    def traced(*a,**k):\n'
        '        identity,argv=command(*a,**k)\n'
        '        return identity,child_command(argv,'+repr(trace_source)+')\n'
        '    _xgpu_fleet.worker_command=traced\n'
        '    return _xgpu_native_main()\n'
        '_xgpu_fleet.main=_xgpu_main\n'
        'import runpy as _xgpu_runpy\n'
        '_xgpu_original_run_module=_xgpu_runpy.run_module\n'
        'def _xgpu_run_module(name,*a,**k):\n'
        '    if name=="exp.r17_integration.ops.formal_workers":return _xgpu_fleet.main()\n'
        '    return _xgpu_original_run_module(name,*a,**k)\n'
        '_xgpu_runpy.run_module=_xgpu_run_module\n')
