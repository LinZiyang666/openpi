"""Evidence-bounded coverage for every numbered INVENTORY capability."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

STATUS = {
    "S01": ("reuse tested", "P1/P2/P3/P4/P5", "CPU fake policy baseline/stock WS; real model G2 pending"),
    "S02": ("reuse tested", "P1/P2/P3/P4/P5", "two actual WS clients; same-bundle CPU threads use connection-local CaptureFactory sinks"),
    "S09": ("reuse tested", "P1", "actual WS YAML registration; native artifact admission checked separately"),
    "S10": ("reuse tested", "P1", "actual WS separate per-client bundle bindings"),
    "S11": ("reuse tested", "P1/P2", "stock lifecycle regression and own WS disconnect; real renderer pending"),
    "S12": ("hook implemented", "P6 + model owners", "sealed binding/content SHA tests; no checkpoint loader qualification"),
    "P05": ("hook implemented", "P1/P6", "P1 array math fit/reload with exact action/donor parity"),
    "P06": ("pending owner", "P1", "guard callback captured; no exhaustive P6 guard equivalence receipt"),
    "P07": ("hook implemented", "P1/P6", "shared corrector fit, whole-episode LOEO, >=3 episodes/task, gripper preservation"),
    "P08": ("pending owner", "P1", "knob/guard replay metadata; periodic gate belongs to P1"),
    "P11": ("reuse tested", "P1/P2 + model owners", "fake actual-cursor tail with zero forward; pi-MW H5 cap retained"),
    "D02": ("hook implemented", "P6 + model owners", "stock DataCollector H5 read via model-owned feature callback; source attestation required"),
    "D03": ("pending owner", "P1 + model owners", "shared StepTrace forwarding seam exists; full native twins/bundle runtime unqualified"),
    "D04": ("reuse tested", "P6", "BlockWriter drain/error/partial/torn-line tests; shared H5 shutdown still model-owner gate"),
    "D05": ("reuse tested", "P6/P2", "shared PerStepRecorder regression; accepted-attempt/manifest join tested"),
    "D06": ("hook implemented", "P6/P1/P2/P5", "core-aware server/isolated worker capture; actual sampler and full identity; terminal ack tested on CPU"),
    "D08": ("reuse tested", "P6 + model owners", "shared BlockWriter, actual two-client WS debug parity, P4 actual saved CPU products"),
    "D09": ("hook implemented", "P6/P2", "sampler-bound native float64 arrays and full model/physical identity; real MT50 semantics pending P2"),
    "D11": ("reuse tested", "P6/P2", "shared stream offset/ACK/spill/fence regression; native file-name adapter tested"),
    "D12": ("pending owner", "P1 + model owners", "three warmup/teacher recipes separated; factor calibration execution unqualified"),
    "D13": ("hook implemented", "P6", "dynamic key fields/action widths -> stock backend entries; native warm explicitly disabled"),
    "D14": ("hook implemented", "P6", "READY dense/ragged memmaps, source SHA, active channels, checksum rejection"),
    "D15": ("hook implemented", "P6/P1", "library read allowlist, nested episode quotas, source split/LOEO tested"),
    "O01": ("hook implemented", "P6/P1", "fit/reset/query interface bridge; frozen harness unchanged"),
    "O02": ("hook implemented", "P6/P1", "active-channel motion/gripper offline action scores and injected sweep; risk/oracle advanced tools pending"),
    "O03": ("reuse tested", "P6/P1", "exact fit reload and generic replay; real WS staged actions/debug bytes compared"),
    "C02": ("hook implemented", "P2/P6", "endpoint compatibility helper tested; scheduler qualification P2"),
    "C03": ("hook implemented", "P2/P6", "done/failed/error/late accepted-join and sealed manifests; driver gate P2"),
    "C06": ("hook implemented", "P2/P6", "all YAML/model/norm/layout/sampler/fit/library/protocol/split/code changes fail audit"),
    "W03": ("hook implemented", "P6/P2", "actual published P2 worker payload imported in isolated Python3.11; no model islands"),
    "W04": ("hook implemented", "P6", "explicit lane + bounded exit75 retry, held free ports and owned stop mocked; no deployment"),
    "W05": ("reuse tested", "P2/P6", "existing EGL build/close lock CPU fake tests; real capacity/GPU renderer pending"),
    "W06": ("hook implemented", "P6", "quiescent lock, fsync/hash snapshot, verify-before-install, idempotence/busy/tamper tests"),
    "A01": ("hook implemented", "P6/P1/P2/coordinator", "real P1/P2 CPU wall ledger/final ack consumed; capture timing excluded from pricing; GPU costs/G4 pending"),
    "A03": ("hook implemented", "P6", "dynamic widths/gripper/units/schedules -> JSON/CSV/PNG/HTML; advanced forensic analyzers pending"),
    "A04": ("hook implemented", "P6/P2", "native raw cards; absent goal/grasp/drop predicates explicit unsupported"),
    "X01": ("hook implemented", "P6/P1 + model owners", "config-only CPU blueprint and short Unix path tests; split execution unqualified"),
}


def generate(inventory, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in Path(inventory).read_text().splitlines():
        match = re.match(r"\| ([A-Z][0-9]{2}) ([^|]+)\| ([^|]+)\|", line)
        if not match:
            continue
        ident, title, classification = match.groups()
        status, owner, scope = STATUS.get(ident, ("pending owner", "P1 + P2/P3/P4/P5", "no P6 passing receipt for entire capability; see owner G0/G1/G2 handback"))
        rows.append(dict(id=ident, feature=title.strip(), classification=classification.split(":", 1)[0].strip(),
                         status=status, gpu_qualified=False, pending_owner=owner, evidence_scope=scope))
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("missing or duplicate inventory capabilities")
    applicability = {
        "single_real_camera_second_camera_ablation": dict(status="not_applicable", reason="MW has one real view; cannot fabricate a wrist"),
        "pi05_mw_formal_commit10": dict(status="not_applicable", reason="native H5 and replan5 only permit commit<=5; a different protocol needs separate certification"),
        "groot_k1_warm_snapshot": dict(status="not_applicable", reason="one-step loop has no interior recoverable point"),
        "dp_k1_formal_descent": dict(status="pending owner", owner="P4/coordinator", reason="K1 remains a lossy research profile; trailing DDIM math does not qualify quality/descent"),
        "cosmos_mw": dict(status="pending owner", owner="coordinator/D", reason="current checkpoint unusable; not an MT50 comparison target"),
        "new_model_split": dict(status="pending owner", owner="P1/model owners/coordinator", reason="CPU blueprint/path only; existing advanced restrictions retained"),
    }
    (directory / "coverage.json").write_text(json.dumps(dict(schema="r17.coverage.v1", gpu_executed=False,
                                                            applicability=applicability, rows=rows), indent=2) + "\n")
    with (directory / "coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
