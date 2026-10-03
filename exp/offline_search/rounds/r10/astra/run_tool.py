"""Run only the standard local emitter, CPU prefit/selftest, or control plan."""
import os
from pathlib import Path
import sys
from .boundary import HERE, install


def main():
    tmp = HERE / 'tmp'
    tmp.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(tmp)
    install()
    mode, *args = sys.argv[1:]
    if mode == 'prefit':
        from exp.offline_search.closed_loop import plugin
        plugin._git_head = lambda: None
        sys.argv = ['prefit', *args]
        raise SystemExit(plugin.prefit_main())
    elif mode == 'emit':
        from exp.offline_search.closed_loop.ops.emit_arms import main
        main(args)
    elif mode == 'plan':
        from exp.offline_search.closed_loop.ops.h100.control import main
        sys.argv = ['control', 'plan', *args]
        main()
    elif mode == 'selftest':
        from exp.offline_search.rounds.r10.selftest import run_test
        run_test(Path(args[0]), args[1])
    else:
        raise ValueError('only local CPU preparation tools are allowed')


if __name__ == '__main__':
    main()
