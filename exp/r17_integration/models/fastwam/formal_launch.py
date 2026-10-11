"""Compatibility entry for the normal, matched-control formal launcher.

The public entry delegates to ops.formal_eval, including H100 and the matched
control gate. Historical launcher_hooks remain solely as a regression fixture.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from .batch_runtime import BatchConfig


@contextmanager
def launcher_hooks(launcher, config):
    original_prepare, original_python=launcher.prepare, launcher.host_python
    def prepare(campaign,directory):
        campaign['fastwam_batching']=config.metadata()
        return original_prepare(campaign,directory)
    def host_python(campaign):
        if campaign['model']=='fastwam' and campaign['domain']=='libero':
            if campaign['server_node']!='local':
                raise ValueError('P3 shim is local-only; H100 requires the P6 launcher seam and qualification')
            return str(Path(__file__).with_name('formal_python.sh'))
        return original_python(campaign)
    launcher.prepare,launcher.host_python=prepare,host_python
    try:
        yield
    finally:
        launcher.prepare,launcher.host_python=original_prepare,original_python


def main(argv=None):
    from exp.r17_integration.ops import formal_eval
    return formal_eval.main(argv)


if __name__=='__main__':
    main()
