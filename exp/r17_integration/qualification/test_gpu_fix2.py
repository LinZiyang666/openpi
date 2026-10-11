"""Q's first-only probe regression must pass with strict reuse admission."""
import json
from pathlib import Path


def test_deployed_probe_first_second_terminal_and_strict_rejections(tmp_path):
    from .probe_pull_replay import replay
    probe=json.loads((Path(__file__).parent/'preflight/gpu_fix1_timan107_actual_probe.json').read_text())['probe']
    result=replay(probe,tmp_path)
    assert result['status']=='PASS' and result['gpu_executed'] is False
    assert [r['probe_present'] for r in result['results']]==[True,False,False]
    assert len(result['strict_rejections'])==7
    assert result['census']['cpu-worker']['pulls']==3
