"""Run the unchanged installed selftest with the server's wrist options.

Only this test process is adapted: the fake key builder exposes the exact K3
layout (zero camera 0, real camera 1); no model or tower is loaded. Shared source
is not modified. The production stage parser/validator and startup hook run.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

def install_hooks():
    from exp.offline_search.closed_loop import plugin, selftest, stage_overrides
    parse = plugin.parse_cli
    def wrist_parse(argv):
        stage, remaining = stage_overrides.parse_flags(list(argv)+['--os-stage1-mode', 'wrist_only'])
        opts, rest = parse(remaining+['--os-tokens', 'off', '--os-no-shadow-native'])
        stage_overrides.validate_method(opts, stage.os_stage1_mode)
        opts.os_stage1_mode = stage.os_stage1_mode
        return opts, rest
    plugin.parse_cli = wrist_parse
    stage_overrides.install_startup_hook(plugin, stage1_mode='wrist_only', miss_steps=10)
    def wrist_build(self, checkpoint_id):
        import torch
        wrist = torch.from_numpy(np.array(self.qc.key_v1[self.row]))
        return {'vision_0': torch.zeros_like(wrist), 'vision_1': wrist,
                'robot_state': torch.from_numpy(np.array(self.qc.rs[self.row]))}
    selftest.FakeKB.build = wrist_build
    def no_tokens(self):
        raise AssertionError('wrist test must not request tokens')
    selftest.FakeKB._slice = no_tokens

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--os-stage1-mode', required=True, choices=['wrist_only'])
    p.add_argument('--os-tokens', required=True, choices=['off'])
    p.add_argument('--os-blind', action='store_true', required=True)
    p.add_argument('--os-judge', required=True)
    a, rest = p.parse_known_args()
    install_hooks()
    from exp.offline_search.closed_loop import selftest
    rc = selftest.main(rest+['--blind', '--judge', a.os_judge, '--no-shadow'])
    out = Path(rest[rest.index('--out')+1])
    logs = [json.loads(line) for line in (out/'decisions_blindtest.jsonl').read_text().splitlines()]
    start = next(r for r in logs if r['ev'] == 'startup')
    assert start['stage1_mode'] == 'wrist_only' and start['miss_steps'] == 10
    assert all(r['stage1_mode'] == 'wrist_only' for r in logs if r['ev'] == 'startup')
    return rc

if __name__ == '__main__':
    sys.exit(main())
