"""Render the retrieval field-weight search into a single self-contained page.

Reads the aggregated per-cell / per-task summaries produced from the ws_search2
arms (both teachers, plus the densify and pinned-PickPlace arms) and writes an
HTML page whose data blob is inlined, so the page needs no network access.

Inputs are the two aggregate JSONs staged by the session (see --ws2 / --pnp);
they are byte-for-byte derivable from the run's `summary_*.json` artifacts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

# ------------------------------------------------------------------
# Task taxonomy
# ------------------------------------------------------------------

# Column order encodes the structural finding: contact manipulation first,
# transport (pick-and-place) second, because the two families behave differently.
CONTACT = [
    ("CloseFridge", "Fridge close"),
    ("OpenStandMixerHead", "Mixer head open"),
    ("OpenDrawer", "Drawer open"),
    ("OpenCabinet", "Cabinet open"),
    ("SlideDishwasherRack", "Dishwasher slide"),
    ("TurnOnSinkFaucet", "Faucet on"),
    ("CloseBlenderLid", "Blender lid close"),
    ("CoffeeSetupMug", "Coffee mug setup"),
]
TRANSPORT = [
    ("PickPlaceToasterToCounter", "Toaster → counter"),
    ("PickPlaceDrawerToCounter", "Drawer → counter"),
    ("PickPlaceCounterToStove", "Counter → stove"),
    ("PickPlaceSinkToCounter", "Sink → counter"),
    ("PickPlaceCounterToCabinet", "Counter → cabinet"),
]
TASK_ORDER = [t for t, _ in CONTACT + TRANSPORT]
TASK_LABEL = dict(CONTACT + TRANSPORT)
N_CONTACT = len(CONTACT)

FIELDS = ["vision_0", "vision_1", "vision_2", "robot_state"]
FIELD_LABEL = ["vision_0", "vision_1", "vision_2", "robot_state"]


def parse_cell(cid: str) -> tuple[str, list[int]]:
    """Return (family, normalized percent weights over FIELDS) for a cell id."""
    if cid.startswith("iso_"):
        w = [100 if f == cid[4:] else 0 for f in FIELDS]
        return "iso", w
    family = cid.split("_vision")[0].split("_robot")[0]
    raw = {f: 0 for f in FIELDS}
    for field, value in re.findall(r"(vision_[012]|robot_state)@(\d+)", cid):
        raw[field] = int(value)
    total = sum(raw.values()) or 1
    # The cid quantises each weight to a 12/25/37/... ladder, so the printed
    # numbers can sum to 98 rather than 100; renormalise for display only.
    return family, [round(raw[f] * 100 / total) for f in FIELDS]


def pack_arm(cells: dict) -> dict:
    """Compress one arm into row tuples plus its per-task totals."""
    rows = []
    per_task = {t: [0, 0] for t in TASK_ORDER}
    for cid, cell in cells.items():
        family, weights = parse_cell(cid)
        succ, denom = [], []
        for task in TASK_ORDER:
            entry = cell["tasks"].get(task)
            if entry is None:
                succ.append(0)
                denom.append(0)
                continue
            succ.append(entry[0])
            denom.append(entry[1])
            per_task[task][0] += entry[0]
            per_task[task][1] += entry[1]
        rows.append([cid, family, weights, succ, denom[0] if denom else 0, cell["macro_sr"]])
    rows.sort(key=lambda r: -r[5])
    return {
        "rows": rows,
        "per_task": {t: per_task[t] for t in TASK_ORDER if per_task[t][1]},
    }


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ws2", required=True, help="aggregate JSON for the ws2/ws2c/ws2e arms")
    ap.add_argument("--pnp", required=True, help="aggregate JSON for the pinned-PickPlace arm")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ws2 = json.loads(pathlib.Path(args.ws2).read_text())["arms"]
    pnp = json.loads(pathlib.Path(args.pnp).read_text())

    data = {
        "tasks": TASK_ORDER,
        "taskLabels": [TASK_LABEL[t] for t in TASK_ORDER],
        "nContact": N_CONTACT,
        "fields": FIELD_LABEL,
        "arms": {
            "groot_tp": pack_arm(ws2["groot_tp/ws2"]["cells"]),
            "pi05": pack_arm(ws2["pi05/ws2"]["cells"]),
        },
        "densify": pack_arm(ws2["groot_tp/ws2e"]["cells"]),
        "control": pack_arm(ws2["groot_tp/ws2c"]["cells"]),
    }

    # Cross-teacher rank agreement over the shared 132 configurations.
    g = {r[0]: r[5] for r in data["arms"]["groot_tp"]["rows"]}
    p = {r[0]: r[5] for r in data["arms"]["pi05"]["rows"]}
    shared = sorted(set(g) & set(p))
    data["rho"] = round(spearman([g[c] for c in shared], [p[c] for c in shared]), 3)
    data["nShared"] = len(shared)

    # Pinned-PickPlace arm keeps its own five-task axis.
    pnp_tasks = sorted({t for c in pnp["pnp_cells"]["ws2"].values() for t in c["tasks"]})
    pnp_rows = []
    for cid, cell in pnp["pnp_cells"]["ws2"].items():
        family, weights = parse_cell(cid)
        pnp_rows.append([
            cid, family, weights,
            [cell["tasks"][t][0] for t in pnp_tasks],
            cell["tasks"][pnp_tasks[0]][1],
            cell["macro_sr"],
        ])
    pnp_rows.sort(key=lambda r: -r[5])
    data["pnp"] = {
        "tasks": pnp_tasks,
        "rows": pnp_rows,
        "teacher": {t: pnp["pnp_teacher"]["tasks"][t]["sr"] for t in pnp_tasks},
        "teacherMacro": pnp["pnp_teacher"]["macro_sr"],
    }

    html = pathlib.Path(args.template).read_text()
    out = pathlib.Path(args.out)
    out.write_text(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
