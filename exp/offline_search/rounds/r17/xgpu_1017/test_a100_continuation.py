import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
import a100_after_pi


class ContinuationTest(unittest.TestCase):
    def test_requires_both_complete_and_mps_closed(self):
        root=Path(__file__).parent/'test_artifacts'/('continuation_'+uuid.uuid4().hex)
        first=root/'results/pi';second=root/'results/pi_cold2'
        first.mkdir(parents=True);second.mkdir()
        complete=dict(status='COMPLETE',mps_cleanup=dict(daemon_closed=True))
        with patch.object(a100_after_pi,'WORK',root):
            (first/'RUN.json').write_text(json.dumps(complete))
            self.assertIsNone(a100_after_pi.terminal_pair('pi'))
            (second/'RUN.json').write_text(json.dumps(dict(status='COMPLETE',mps_cleanup={})))
            self.assertIsNone(a100_after_pi.terminal_pair('pi'))
            (second/'RUN.json').write_text(json.dumps(complete))
            self.assertEqual(a100_after_pi.terminal_pair('pi'),second)

    def test_failed_capacity_run_stops_before_cleanup(self):
        root=Path(__file__).parent/'test_artifacts'/('continuation_'+uuid.uuid4().hex)
        first=root/'results/pi';first.mkdir(parents=True)
        (first/'RUN.json').write_text(json.dumps(dict(status='FAILED',mps_cleanup=dict(daemon_closed=True))))
        with patch.object(a100_after_pi,'WORK',root),self.assertRaisesRegex(RuntimeError,'retain its files'):
            a100_after_pi.terminal_pair('pi')


if __name__=='__main__':unittest.main()
