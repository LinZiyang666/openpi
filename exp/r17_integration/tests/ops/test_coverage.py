import re
from pathlib import Path

from exp.r17_integration.ops.coverage import generate


def test_coverage_has_every_inventory_id_and_no_gpu_claim(tmp_path):
    inventory = Path(__file__).resolve().parents[2] / "INVENTORY.md"
    rows = generate(inventory, tmp_path)
    expected = set(re.findall(r"^\| ([A-Z][0-9]{2}) ", inventory.read_text(), flags=re.M))
    assert {r["id"] for r in rows} == expected
    assert all(not r["gpu_qualified"] and r["pending_owner"] for r in rows)
    assert all(r["status"] in ("reuse tested", "hook implemented", "pending owner", "not_applicable") for r in rows)
