"""Model-neutral fit/reset/query replay and resident/transport comparison."""
from __future__ import annotations

import numpy as np

from exp.r17_integration.debug.validate import canonical, numeric_arrays

RESULT_FIELDS = ("donor_ids", "weights", "guard", "knob", "stage_calls", "actual_nfe", "cost_ms")


def replay(method, library, episodes):
    method.fit(library)
    results = []
    for episode, queries in episodes:
        method.reset(episode)
        for query in queries:
            output = method.query(query)
            missing = [key for key in RESULT_FIELDS + ("actions",) if key not in output]
            if missing:
                raise ValueError(f"incomplete replay result: {missing}")
            results.append(dict(metadata={k: output[k] for k in RESULT_FIELDS},
                                arrays=numeric_arrays({k: output[k] for k in ("actions", "normalized_actions") if k in output})))
    return results


def compare_replays(reference, candidate, *, rtol=0., atol=0.):
    if len(reference) != len(candidate):
        raise AssertionError("replay count mismatch")
    for i, (a, b) in enumerate(zip(reference, candidate)):
        if canonical(a["metadata"]) != canonical(b["metadata"]):
            raise AssertionError(f"decision {i}: donor/guard/knob/cost mismatch")
        if set(a["arrays"]) != set(b["arrays"]):
            raise AssertionError("replay array fields mismatch")
        for name, value in a["arrays"].items():
            other = b["arrays"][name]
            if value.shape != other.shape or value.dtype != other.dtype:
                raise AssertionError(f"decision {i}: array geometry mismatch")
            if rtol == 0 and atol == 0:
                if value.tobytes() != other.tobytes():
                    raise AssertionError(f"decision {i}: action bytes mismatch")
            else:
                np.testing.assert_allclose(value, other, rtol=rtol, atol=atol)
    return dict(status="passed", decisions=len(reference), rtol=rtol, atol=atol)
