"""Static review artifacts from the same T1/T2 calculations as the CSVs."""

from __future__ import annotations

from collections import Counter
import html
from pathlib import Path

import numpy as np

from .adapters import EnvironmentAdapter
from .common import Unavailable, fingerprint
from .forensics import analyse


STAGES = [
    "settle",
    "approach",
    "grasp_window",
    "carry",
    "place",
    "release",
    "retreat",
    "fixture",
    "unknown",
]
SOURCES = [
    "settle",
    "cache",
    "cache_tail",
    "follow",
    "policy",
    "policy_tail",
    "unknown",
]


def plotting():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/r8_S6/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    return pyplot


def html_document(title, content):
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title>"
        + "<style>body{font:16px sans-serif;max-width:1150px;margin:2em auto;padding:1em}img{max-width:100%}table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #bbb}pre{white-space:pre-wrap}code{font-size:13px}</style></head><body><h1>"
        + html.escape(title)
        + "</h1>"
        + content
        + "</body></html>"
    )


def table_html(rows, fields):
    result = (
        "<table><tr>"
        + "".join("<th>" + html.escape(k) + "</th>" for k in fields)
        + "</tr>"
    )
    for row in rows:
        result += (
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(row.get(k, "unavailable"))) + "</td>"
                for k in fields
            )
            + "</tr>"
        )
    return result + "</table>"


def card(episode, out, config=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "episode_" + fingerprint([episode.meta["arm"], episode.key])[:20]
    title = "%s task %s / init %s / attempt %s" % (
        episode.meta["arm"],
        episode.meta["task_id"],
        episode.meta["init"],
        episode.meta["attempt"],
    )
    try:
        forensic, data = analyse(episode, config)
        adapter = EnvironmentAdapter(episode, config)
    except (Unavailable, KeyError, ValueError) as exc:
        plt = plotting()
        fig, axis = plt.subplots(figsize=(9, 3))
        axis.axis("off")
        axis.text(
            0.02,
            0.8,
            title + "\nPhysical evidence unavailable:\n" + str(exc),
            wrap=True,
            transform=axis.transAxes,
        )
        fig.savefig(out / (stem + ".png"), dpi=110)
        plt.close(fig)
        content = (
            "<p>Unavailable: "
            + html.escape(str(exc))
            + "</p><img src='"
            + stem
            + ".png' alt='unavailable physical evidence'>"
        )
        (out / (stem + ".html")).write_text(html_document(title, content))
        return dict(
            episode.identity,
            status="unavailable",
            reason=str(exc),
            html=stem + ".html",
            png=stem + ".png",
        )
    plt = plotting()
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 11),
        gridspec_kw={"height_ratios": [1.2, 3, 2, 2]},
        constrained_layout=True,
    )
    source = [
        episode.decision_at(i).get(
            "src", "settle" if episode.controls["is_settle"][i] else "unknown"
        )
        for i in range(episode.n)
    ]
    strip = np.array(
        [
            [SOURCES.index(x) if x in SOURCES else len(SOURCES) - 1 for x in source],
            [STAGES.index(str(x)) for x in data["stage"]],
        ],
        float,
    )
    axes[0].imshow(
        strip,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        extent=[0, episode.n, 1.5, -0.5],
        cmap="tab10",
        vmin=-0.5,
        vmax=9.5,
    )
    axes[0].set_yticks([0, 1], ["served source", "sim truth"])
    axes[0].set_title("%s: %s (automatic heuristic)" % (title, forensic["label"]))
    from matplotlib.patches import Patch

    colors = plt.get_cmap("tab10")
    handles = [
        Patch(facecolor=colors(i), label="src: " + name)
        for i, name in enumerate(SOURCES)
        if name in source
    ]
    handles += [
        Patch(facecolor=colors(i), label="truth: " + name)
        for i, name in enumerate(STAGES)
        if name in data["stage"]
    ]
    axes[0].legend(
        handles=handles,
        fontsize=6,
        ncol=4,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.2),
    )
    # Display actual logged triggers only; absence remains unavailable.
    trigger_count = 0
    for d in episode.decisions:
        diag = d.get("diag", {}) or {}
        if any(
            bool(diag.get(k))
            for k in (
                "stall",
                "state_valve",
                "deviation_entry",
                "event_entry",
                "os_call_stall",
            )
        ):
            axes[0].axvline(episode.before_index(d) + 1, color="red", alpha=0.5)
            trigger_count += 1
    eef = episode.controls["eef_pos"]
    axes[1].plot(eef[:, 0], eef[:, 1], label="eef")
    for k, name in enumerate(adapter.names):
        obj = episode.controls["obj_pos"][:, k]
        axes[1].plot(obj[:, 0], obj[:, 1], label=name, alpha=0.7)
        axes[2].plot(np.arange(episode.n), obj[:, 2], label=name)
    axes[1].set(xlabel="world X", ylabel="world Y", aspect="equal")
    axes[1].legend(fontsize=7, ncol=3)
    axes[2].set(ylabel="object world Z", xlabel="actual control")
    width = [adapter.width(i) for i in range(episode.n)]
    if all(x is not None for x in width):
        axes[3].plot(np.arange(episode.n), width, label="measured aperture")
    else:
        axes[3].text(
            0.01, 0.9, "measured width unavailable", transform=axes[3].transAxes
        )
    command_axis = axes[3].twinx()
    command_axis.plot(
        np.arange(episode.n),
        episode.controls["action"][:, int(adapter.setting("gripper_dim"))],
        label="gripper command",
        alpha=0.5,
    )
    axes[3].set(xlabel="actual control", ylabel="measured aperture (native units)")
    command_axis.set_ylabel("issued gripper command")
    width_handles, width_labels = axes[3].get_legend_handles_labels()
    command_handles, command_labels = command_axis.get_legend_handles_labels()
    axes[3].legend(
        width_handles + command_handles, width_labels + command_labels, fontsize=8
    )
    if forensic["onset_control"] is not None:
        for axis in (axes[0], axes[2], axes[3]):
            axis.axvline(
                forensic["onset_control"],
                color="black",
                linestyle="--",
                label="heuristic onset",
            )
    fig.savefig(out / (stem + ".png"), dpi=110)
    plt.close(fig)
    content = (
        "<p>Automatic labels require human validation. Post-settle reference: "
        + html.escape(str(forensic["reference_control"]))
        + ". Onset: "
        + html.escape(str(forensic["onset_control"]))
        + ". Logged trigger markers: "
        + str(trigger_count)
        + ".</p><img src='"
        + stem
        + ".png' alt='physical episode timeline'>"
    )
    content += (
        "<p>Source palette: "
        + html.escape(", ".join(SOURCES))
        + ". Truth palette: "
        + html.escape(", ".join(STAGES))
        + ".</p>"
    )
    content += table_html(
        forensic["predicate_detail"],
        [
            "predicate",
            "object",
            "destination",
            "label",
            "first_near",
            "first_lift",
            "onset_control",
            "final",
        ],
    )
    content += table_html(
        episode.decisions,
        [
            "decision_seq",
            "src",
            "vision",
            "look_reason",
            "miss_reason",
            "blind_age_controls",
        ],
    )
    images_status = "unavailable"
    if episode.reader_arm is not None and episode.decisions:
        decision = (
            episode.decision_at(forensic["onset_control"])
            if forensic["onset_control"] is not None
            else episode.decisions[-1]
        )
        if decision.get("decision_id"):
            try:
                from PIL import Image

                frames = episode.reader_arm.images([decision["decision_id"]])
                for camera, arr in frames.items():
                    image_name = stem + "_camera_" + fingerprint(camera)[:8] + ".png"
                    Image.fromarray(np.asarray(arr[0], np.uint8)).save(out / image_name)
                    content += (
                        "<p>"
                        + html.escape(camera)
                        + " at decision "
                        + html.escape(str(decision["decision_seq"]))
                        + "</p><img src='"
                        + image_name
                        + "' alt='captured wire image'>"
                    )
                images_status = "available" if frames else "unavailable"
            except (KeyError, ValueError, OSError) as exc:
                content += "<p>Images unavailable: " + html.escape(str(exc)) + "</p>"
    (out / (stem + ".html")).write_text(html_document(title, content))
    return dict(
        episode.identity,
        status="available",
        label=forensic["label"],
        onset_control=forensic["onset_control"],
        html=stem + ".html",
        png=stem + ".png",
        images_status=images_status,
    )


def rollup(episodes, out, config=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, labels, stages = [], Counter(), Counter()
    for ep in episodes:
        try:
            row, data = analyse(ep, config)
            labels[row["label"]] += 1
            stages.update(str(x) for x in data["stage"] if x != "settle")
            rows.append(row)
        except (Unavailable, ValueError, KeyError) as exc:
            rows.append(dict(ep.identity, status="unavailable", reason=str(exc)))
    name = str(episodes[0].meta["arm"]) if episodes else "empty"
    stem = "arm_" + fingerprint(name)[:20]
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, counts, title in (
        (axes[0], labels, "Episode labels"),
        (axes[1], stages, "Active control truth-stage exposure"),
    ):
        names = sorted(counts)
        axis.barh(names, [counts[n] for n in names])
        axis.set_title(title)
        axis.set_xlabel("count")
    fig.suptitle(
        "%s: %d accepted episodes, %d physically available"
        % (name, len(episodes), sum(r["status"] == "available" for r in rows))
    )
    fig.savefig(out / (stem + ".png"), dpi=110)
    plt.close(fig)
    content = (
        "<p>Accepted episodes: %d. Labels are unvalidated heuristics. Stage denominator: %d active controls.</p>"
        % (len(episodes), sum(stages.values()))
    )
    content += "<img src='" + stem + ".png' alt='arm physical rollup'>"
    content += table_html(
        rows,
        [
            "task_id",
            "init",
            "attempt",
            "success",
            "status",
            "label",
            "onset_control",
            "tail_motion",
        ],
    )
    (out / (stem + ".html")).write_text(html_document(name, content))
    ledger = []
    for ep in episodes:
        try:
            _, data = analyse(ep, config)
        except (Unavailable, ValueError, KeyError):
            continue
        counts = Counter(
            (str(data["stage"][i]), ep.decision_at(i).get("src", "unknown"))
            for i in range(ep.n)
            if data["active"][i]
        )
        ledger += [
            dict(
                ep.identity,
                status="available",
                truth_stage=stage,
                src=src,
                active_controls=count,
            )
            for (stage, src), count in counts.items()
        ]
    return (
        rows,
        ledger,
        {
            "arm": name,
            "accepted_episodes": len(episodes),
            "labels": dict(labels),
            "active_controls": sum(stages.values()),
            "stage_controls": dict(stages),
            "html": stem + ".html",
            "png": stem + ".png",
        },
    )
