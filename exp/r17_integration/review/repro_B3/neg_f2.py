import sys
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import pytest
from exp.r17_integration.models.dp import protocols
from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
protocols.libero_spec_from_document = lambda document, *, replan_steps: LiberoBenchmarkSpec(**document)   # round-2 behaviour
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp=/tmp/reviewB3/neg_f2_tmp",
    "/home/weiland/projects/openpi/exp/r17_integration/tests/models/dp/test_dp_review_b3.py"]))
