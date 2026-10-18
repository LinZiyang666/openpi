"""Original R10 control launcher, with all file transfers routed through rsync.

No setup/global runtime replacement is provided. The original H100 installation
and model/worker topology are reused. Only the coordinator invokes sync/chain.
"""
from __future__ import annotations

import sys
import os
from contextlib import nullcontext
from unittest.mock import patch


def rsync_transport(native_run, transfer):
    def run(command,*args,**kwargs):
        command=list(map(str,command))
        if command[:2] in (['tether','push'],['tether','pull']):
            transfer(['rsync',command[1],*command[-2:]])
            return ''
        return native_run(command,*args,**kwargs)
    return run


def main():
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.r17_integration.ops.formal_eval import transfer
    if len(sys.argv)<2 or sys.argv[1] not in ('plan','sync','chain','collect'):
        raise SystemExit('Use plan, sync, chain, or collect; existing runtime setup is preserved.')
    from exp.offline_search.rounds.r17.filter_failed_knob.vla_shared import r15
    scope = os.environ.get('FFK_WORKER_FLEET_ROOT')
    with patch.object(control,'run',rsync_transport(control.run,transfer)), \
            r15(scope) if scope else nullcontext():
        return control.main()


if __name__=='__main__':main()
