import sys
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
import pytest
from exp.r17_integration.models.fastwam import staged
orig = staged.FastWAMAdapter.run_stage1_batch
def eager(self, prepared):            # round-2 behaviour: hash every request in stage1
    for p in prepared:
        staged._scope_digest(p.data["prompt"], p.data["input_image"], p.data["proprio"])
    return orig(self, prepared)
staged.FastWAMAdapter.run_stage1_batch = eager
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp=/tmp/reviewB3/neg_f1_tmp",
    "/home/weiland/projects/openpi/exp/r17_integration/tests/models/fastwam/test_fastwam_review_b3.py",
    "-k", "never_hash"]))
