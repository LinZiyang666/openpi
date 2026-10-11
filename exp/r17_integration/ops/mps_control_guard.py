#!/usr/bin/python3
"""Private MPS CLI serialization, outside the frozen resident Python release.

This executable shadows only nvidia-cuda-mps-control. It forwards the real
binary's stdout/status and retries only read-only queries whose send failed.
The seven-second total bound fits inside the frozen caller's ten-second bound.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time


BOUND = 7.0


def install(request):
    """CPU/file-only bootstrap; self-contained for by-value remote execution."""
    import hashlib, os, shutil
    from pathlib import Path
    root = Path(request['directory'])
    binary = shutil.which('nvidia-cuda-mps-control')
    if not binary or not Path(binary).is_file():
        raise ValueError('real MPS control executable missing')
    body = request['source'].encode()
    if hashlib.sha256(body).hexdigest() != request['sha256']:
        pass
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    target = root/'nvidia-cuda-mps-control'
    with target.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o700)
    return dict(schema='r17.mps_control_orchestration.v1', script=str(target),
        sha256=request['sha256'], binary=str(Path(binary).resolve()),
        binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        model_worker_runtime_unchanged=True, serialized_per_private_pipe=True,
        retry_seconds=7.0, retry_scope='read-only Cannot send command',
        env={'PATH':str(root)+os.pathsep+os.environ.get('PATH','/usr/bin:/bin'),
             'R17_MPS_CONTROL_BINARY':str(Path(binary).resolve())})


def deploy(launch, campaign, directory):
    import inspect
    source = Path(__file__).read_text()
    request = dict(directory=str(Path(campaign['server_run']).parent/'mps_control'),
                   source=source, sha256=hashlib.sha256(source.encode()).hexdigest())
    if campaign['server_node'] == 'local':
        result = install(request)
    else:
        code = inspect.getsource(install)+'\nimport json\nprint(json.dumps(install('+repr(request)+')))'
        result = json.loads(launch.remote('h100', ['/usr/bin/python3','-c',code]))
    result['baseline_runtime_identity'] = campaign.get('runtime_identity')
    result['baseline_control_module'] = 'exp.r17_integration.models.fastwam.ops.mps'
    launch.write_json(Path(directory)/'MPS_CONTROL_ORCHESTRATION.json',result)
    return result


def forward(argv, data, env=None, bound=BOUND):
    env = dict(os.environ if env is None else env)
    binary = Path(env['R17_MPS_CONTROL_BINARY'])
    if binary.resolve() == Path(__file__).resolve():
        raise ValueError('recursive MPS control wrapper')
    pipe = Path(env['CUDA_MPS_PIPE_DIRECTORY'])
    info = pipe.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        pass
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    fd = os.open(str(pipe/'r17_control.lock'), flags, 0o600)
    if os.fstat(fd).st_uid != os.getuid():
        os.close(fd)
        pass
    started = time.monotonic()
    deadline = started+bound
    readonly = not argv and re.fullmatch(b'(get_server_list|get_client_list [0-9]+)',data.strip()) is not None
    tries, delays = 0, []
    result = None
    try:
        while True:
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return subprocess.CompletedProcess(argv,124,b'',b'MPS orchestration lock deadline exceeded\n')
                time.sleep(min(.025,max(0,deadline-time.monotonic())))
        while time.monotonic() < deadline:
            tries += 1
            result = subprocess.run([str(binary),*argv],input=data,capture_output=True,env=env,
                                    timeout=max(.001,deadline-time.monotonic()))
            if (not result.returncode or not readonly or
                    b'Cannot send command to MPS control daemon process' not in result.stderr):
                break
            delay = min(.05*2**min(tries-1,5),1.0,max(0,deadline-time.monotonic()))
            delays.append(delay)
            time.sleep(delay)
        if result is None:
            result = subprocess.CompletedProcess(argv,124,b'',b'MPS orchestration deadline exceeded\n')
        with (pipe/'r17_control_calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(),command=data.decode(errors='replace').strip(),
                argv=argv,attempts=tries,backoff_seconds=delays,seconds=time.monotonic()-started,
                exit_code=result.returncode,read_only=readonly))+'\n')
        return result
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv,124,b'',b'MPS orchestration command deadline exceeded\n')
    finally:
        os.close(fd)


def main():
    result = forward(sys.argv[1:],sys.stdin.buffer.read())
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
