"""Reintroduce the three reviewed mechanisms in this process only.

This is a scoped mutation witness, not a replay of an entire historical tree.
No production files are replaced. Each mutation must break its regression;
the restored implementation must pass the same regression with fresh assets.
"""
import argparse
import json
from pathlib import Path

import pytest

from exp.r17_integration.common.fake_adapter import fake_provider
from exp.r17_integration.common import benchmarks, recipe
from exp.r17_integration.search.store import Store
from exp.r17_integration.tests.common import test_common_fix_round2 as checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=False)
    provider = fake_provider("r17.p1.reviewed_mutation")
    def reopen(path, expected):
        store = Store(path)
        assert store.manifest["content_digest"] == expected
        return store
    def n1(patch):patch.setattr(checks,"target_ir_to_gap",lambda target:(1-target)/target)
    def n2(patch):patch.setattr(benchmarks,"validate_task_table",lambda cfg:None)
    def n4(patch):
        patch.setattr(recipe,"verified_store",reopen)
        patch.setattr(checks,"verified_store",reopen)
    cases = [
        ("A2N1",n1,lambda path,patch:checks.test_n1_shortcut_and_unbound_costs_fail_closed(provider,path)),
        ("A2N2",n2,lambda path,patch:checks.test_n2_canonical_order_and_spec_digest_reject_before_store_access(provider,path,patch,"reverse")),
        ("A2N4",n4,lambda path,patch:checks.test_n4_store_sha_once_per_load_across_connections_and_factories(provider,path,patch)),
    ]
    rows = []
    for finding,mutate,run in cases:
        red = None
        with pytest.MonkeyPatch.context() as patch:
            mutate(patch)
            try:run(args.output_directory/(finding+"_red"),patch)
            except (Exception,pytest.fail.Exception) as exc:red = dict(type=type(exc).__name__,message=str(exc))
        if red is None:raise AssertionError("reviewed mechanism did not break regression: "+finding)
        with pytest.MonkeyPatch.context() as patch:
            run(args.output_directory/(finding+"_green"),patch)
        rows.append(dict(finding=finding,reviewed_mechanism="red",failure=red,restored_implementation="passed"))
    receipt = dict(schema="r17.p1.fix2.scoped_mutation.v1",scope="reviewed mechanisms in process; not historical checkout",
                   gpu_used=False,red=3,green=3,rows=rows)
    (args.output_directory/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("3 reviewed-mechanism failures;3 restored-implementation passes")


if __name__ == "__main__":main()
