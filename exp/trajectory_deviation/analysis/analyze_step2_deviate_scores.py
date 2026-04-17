"""Generate the Step 2 trajectory-deviation analysis report and figures.

The input files are the final Step 2 score JSONs produced by
``exp.trajectory_deviation.compute_deviate_scores``:

    deviate_score_{cfg}.json

Each JSON is keyed by ``task_i/episode_j`` and contains cycle-level
``background_l2``, ``cache_l2``, and ``deviate_score`` arrays.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DEFAULT_CONFIGS = ("clip_w7_d4", "spatial16_w8_d4", "max_pool_w3_d5")
DISPLAY = {
    "clip_w7_d4": "clip",
    "spatial16_w8_d4": "spatial16",
    "max_pool_w3_d5": "max_pool",
}
BANDS = (
    ("<1", 0.0, 1.0, "#2f9e44"),
    ("1-2", 1.0, 2.0, "#228be6"),
    ("2-5", 2.0, 5.0, "#f08c00"),
    (">=5", 5.0, math.inf, "#e03131"),
)


@dataclass(frozen=True)
class EpisodeRecord:
    config: str
    episode: str
    task_id: int
    scores: np.ndarray
    background_l2: np.ndarray
    cache_l2: np.ndarray

    @property
    def cycles(self) -> int:
        return int(self.scores.size)

    @property
    def ge5_count(self) -> int:
        return int(np.sum(self.scores >= 5.0))

    @property
    def ge5_indices(self) -> list[int]:
        return np.flatnonzero(self.scores >= 5.0).astype(int).tolist()


def _score_path(score_dir: Path, cfg: str) -> Path:
    return score_dir / f"deviate_score_{cfg}.json"


def _task_id(episode: str) -> int:
    match = re.match(r"task_(\d+)/episode_\d+$", episode)
    if not match:
        raise ValueError(f"Unexpected episode key: {episode}")
    return int(match.group(1))


def _band_mask(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if math.isinf(hi):
        return values >= lo
    return (values >= lo) & (values < hi)


def _load_episode_records(score_dir: Path, configs: list[str]) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    expected = {"background_l2", "cache_l2", "deviate_score"}
    for cfg in configs:
        path = _score_path(score_dir, cfg)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        for episode, values in sorted(data.items()):
            if set(values) != expected:
                raise ValueError(f"{path}:{episode} has keys {sorted(values)}")
            arrays = {key: np.asarray(values[key], dtype=float) for key in expected}
            lengths = {key: int(arr.size) for key, arr in arrays.items()}
            if len(set(lengths.values())) != 1 or lengths["deviate_score"] == 0:
                raise ValueError(f"{path}:{episode} invalid lengths {lengths}")
            records.append(
                EpisodeRecord(
                    config=cfg,
                    episode=episode,
                    task_id=_task_id(episode),
                    scores=arrays["deviate_score"],
                    background_l2=arrays["background_l2"],
                    cache_l2=arrays["cache_l2"],
                )
            )
    return records


def _read_task_descriptions(gt_dir: Path, task_ids: list[int]) -> dict[int, str]:
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        return {task_id: f"task_{task_id}" for task_id in task_ids}

    descriptions: dict[int, str] = {}
    for task_id in task_ids:
        task_dir = gt_dir / f"task_{task_id}"
        first_h5 = next(iter(sorted(task_dir.glob("episode_*.h5"))), None)
        if first_h5 is None:
            descriptions[task_id] = f"task_{task_id}"
            continue
        with h5py.File(first_h5, "r") as handle:
            task_name = handle.attrs.get("task_name", f"task_{task_id}")
        if isinstance(task_name, bytes):
            task_name = task_name.decode("utf-8")
        descriptions[task_id] = str(task_name)
    return descriptions


def _short_task(text: str) -> str:
    text = text.removeprefix("pick up the black bowl ")
    text = text.removesuffix(" and place it on the plate")
    replacements = {
        "between the plate and the ramekin": "盘子和小模子之间",
        "next to the ramekin": "小模子旁边",
        "from table center": "桌面中央",
        "on the cookie box": "饼干盒上",
        "in the top drawer of the wooden cabinet": "木柜顶层抽屉里",
        "on the ramekin": "小模子上",
        "next to the cookie box": "饼干盒旁边",
        "on the stove": "炉灶上",
        "next to the plate": "盘子旁边",
        "on the wooden cabinet": "木柜上",
    }
    return replacements.get(text, text) + " -> 放到盘子"


def _config_records(records: list[EpisodeRecord], cfg: str) -> list[EpisodeRecord]:
    return [record for record in records if record.config == cfg]


def _flat_scores(records: list[EpisodeRecord]) -> np.ndarray:
    return np.concatenate([record.scores for record in records])


def _overall_rows(records: list[EpisodeRecord], configs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for cfg in configs:
        cfg_records = _config_records(records, cfg)
        scores = _flat_scores(cfg_records)
        ge5_counts = [record.ge5_count for record in cfg_records]
        rows.append(
            {
                "config": cfg,
                "episodes": len(cfg_records),
                "cycles": int(scores.size),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "p90": float(np.percentile(scores, 90)),
                "p95": float(np.percentile(scores, 95)),
                "p99": float(np.percentile(scores, 99)),
                "max": float(np.max(scores)),
                "ge5_ep_mean": float(np.mean(ge5_counts)),
                "zero_ge5_episodes": int(np.sum(np.asarray(ge5_counts) == 0)),
            }
        )
    return rows


def _band_rows(records: list[EpisodeRecord], configs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for cfg in configs:
        scores = _flat_scores(_config_records(records, cfg))
        row = {"config": cfg}
        for label, lo, hi, _ in BANDS:
            count = int(np.sum(_band_mask(scores, lo, hi)))
            row[label] = count
            row[f"{label}_pct"] = float(count / scores.size * 100)
        rows.append(row)
    return rows


def _task_rows(
    records: list[EpisodeRecord],
    task_descriptions: dict[int, str],
) -> list[dict[str, Any]]:
    rows = []
    by_task: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_task[record.task_id].append(record)
    for task_id, task_records in sorted(by_task.items()):
        scores = _flat_scores(task_records)
        episode_counts = [record.ge5_count for record in task_records]
        rows.append(
            {
                "task_id": task_id,
                "description": _short_task(task_descriptions[task_id]),
                "ge5_pct": float(np.mean(scores >= 5.0) * 100),
                "ge5_ep_mean": float(np.mean(episode_counts)),
                "mean": float(np.mean(scores)),
            }
        )
    rows.sort(key=lambda row: row["ge5_pct"], reverse=True)
    return rows


def _top_episode_rows(records: list[EpisodeRecord], limit: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "config": record.config,
                "episode": record.episode,
                "cycles": record.cycles,
                "ge5_count": record.ge5_count,
                "max": float(np.max(record.scores)),
                "indices": record.ge5_indices,
            }
        )
    rows.sort(key=lambda row: (row["ge5_count"], row["max"]), reverse=True)
    return rows[:limit]


def _plot_distribution(records: list[EpisodeRecord], configs: list[str], out_path: Path) -> None:
    values_by_cfg = {cfg: _flat_scores(_config_records(records, cfg)) for cfg in configs}
    xmax = max(float(np.percentile(values, 99.7)) for values in values_by_cfg.values())
    xmax = max(8.0, min(30.0, math.ceil(xmax)))
    bin_edges = np.linspace(0.0, xmax, 121)

    hists = {}
    max_density = 0.0
    for cfg, values in values_by_cfg.items():
        hist, edges = np.histogram(values, bins=bin_edges, density=True)
        hists[cfg] = (hist, edges)
        max_density = max(max_density, float(hist.max()))

    fig, axes = plt.subplots(
        len(configs),
        1,
        figsize=(12, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    for ax, cfg in zip(axes, configs):
        values = values_by_cfg[cfg]
        hist, edges = hists[cfg]
        widths = np.diff(edges)
        colors = []
        for center in edges[:-1] + widths / 2:
            for _, lo, hi, color in BANDS:
                if center >= lo and (math.isinf(hi) or center < hi):
                    colors.append(color)
                    break
        ax.bar(edges[:-1], hist, width=widths, align="edge", color=colors, edgecolor=colors)
        for threshold in (1.0, 2.0, 5.0):
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.65)
            ax.text(
                threshold,
                max_density * 0.92,
                f"{threshold:g}",
                ha="center",
                va="top",
                fontsize=9,
                backgroundcolor="white",
            )
        ax.set_title(
            f"{cfg}: n={len(values)}, mean={np.mean(values):.2f}, "
            f"median={np.median(values):.2f}, max={np.max(values):.2f}"
        )
        ax.set_ylabel("Density")
        ax.set_ylim(0, max_density * 1.08)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("deviate_score = cache_l2 / max(background_l2, floor)")
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor=color, label=label) for label, _, _, color in BANDS],
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_task_heatmap(records: list[EpisodeRecord], configs: list[str], out_path: Path) -> None:
    task_ids = sorted({record.task_id for record in records})
    values = np.zeros((len(configs), len(task_ids)), dtype=float)
    for i, cfg in enumerate(configs):
        for j, task_id in enumerate(task_ids):
            task_records = [
                record for record in records if record.config == cfg and record.task_id == task_id
            ]
            scores = _flat_scores(task_records)
            values[i, j] = float(np.mean(scores >= 5.0) * 100)

    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    image = ax.imshow(values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(task_ids)), [f"task_{task_id}" for task_id in task_ids])
    ax.set_yticks(range(len(configs)), [DISPLAY.get(cfg, cfg) for cfg in configs])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_title("Cycle share with deviate_score >= 5 by task and keybuilder")
    fig.colorbar(image, ax=ax, label="% cycles >= 5")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_episode_hist(records: list[EpisodeRecord], configs: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    max_count = max(record.ge5_count for record in records)
    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    for cfg in configs:
        counts = [record.ge5_count for record in _config_records(records, cfg)]
        ax.hist(counts, bins=bins, alpha=0.45, label=DISPLAY.get(cfg, cfg))
    ax.set_xlabel("# cycles with deviate_score >= 5 in one episode")
    ax.set_ylabel("Episode count")
    ax.set_title("Per-episode count of very-large deviation points")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_relative_positions(records: list[EpisodeRecord], configs: list[str], out_path: Path) -> None:
    bins = np.linspace(0.0, 1.0, 6)
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for offset, cfg in zip((-width, 0.0, width), configs):
        positions = []
        for record in _config_records(records, cfg):
            denom = max(record.cycles - 1, 1)
            positions.extend(index / denom for index in record.ge5_indices)
        hist, _ = np.histogram(positions, bins=bins)
        ax.bar(x + offset, hist, width=width, label=DISPLAY.get(cfg, cfg))
    ax.set_xticks(x, labels)
    ax.set_ylabel("# cycles with deviate_score >= 5")
    ax.set_title("Where very-large deviation points appear inside episodes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_task_ranking(task_rows: list[dict[str, Any]], out_path: Path) -> None:
    rows = list(reversed(task_rows))
    labels = [f"task_{row['task_id']}" for row in rows]
    values = [row["ge5_pct"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    bars = ax.barh(labels, values, color="#f08c00")
    for bar, value in zip(bars, values):
        ax.text(value + 0.15, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    ax.set_xlabel("% cycles with deviate_score >= 5")
    ax.set_title("Pooled task risk ranking across all keybuilders")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_top_episodes(top_rows: list[dict[str, Any]], out_path: Path) -> None:
    rows = list(reversed(top_rows))
    labels = [f"{DISPLAY.get(row['config'], row['config'])} {row['episode']}" for row in rows]
    values = [row["ge5_count"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    bars = ax.barh(labels, values, color="#e03131")
    for bar, row in zip(bars, rows):
        ax.text(
            row["ge5_count"] + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"max={row['max']:.1f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("# cycles with deviate_score >= 5")
    ax.set_title("Episodes with the most very-large deviation points")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _write_report_en(
    out_path: Path,
    score_dir: Path,
    configs: list[str],
    overall_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# Step 2 Deviate Score Analysis",
        "",
        f"Data source: `{score_dir}/deviate_score_{{cfg}}.json`.",
        "",
        "Generated by:",
        "",
        "```bash",
        "uv run python -m exp.trajectory_deviation.analysis.analyze_step2_deviate_scores \\",
        f"    --score-dir {score_dir} \\",
        "    --gt-dir exp/trajectory_deviation/data/gt",
        "```",
        "",
        "The Chinese version can be generated by:",
        "",
        "```bash",
        "uv run python -m exp.trajectory_deviation.analysis.analyze_step2_deviate_scores \\",
        f"    --score-dir {score_dir} \\",
        "    --gt-dir exp/trajectory_deviation/data/gt \\",
        f"    --out-md {score_dir}/step2_deviate_score_analysis_zh.md \\",
        "    --language zh",
        "```",
        "",
        "The distribution-only plotting helper is:",
        "",
        "```bash",
        "uv run python -m exp.trajectory_deviation.analysis.plot_deviate_score_distribution \\",
        f"    --score-dir {score_dir}",
        "```",
        "",
        "This analysis follows the Step 1a report style, but uses the Step 2 deviate-score outputs. "
        "A cycle-level score is computed as `cache_l2 / max(background_l2, 0.1)`. Higher values "
        "mean the cache replay action is farther from GT than ordinary pure-inference variation.",
        "",
        "## Executive Summary",
        "",
        "- The three keybuilders have very similar global distributions: median score is about "
        "`1.82-1.86`, and `>=5` very-large points account for about `6.2-6.7%` of cycles.",
        "- The typical episode has about `21-22` replan cycles and about `1.3-1.5` cycles with "
        "score `>=5`. The median episode has exactly `1` very-large point.",
        "- High-risk points are not uniformly distributed: they are most common in the middle of "
        "the trajectory and appear again near the end. The first 20% of each episode has the "
        "fewest very-large points.",
        "- Task-level effects are stronger than keybuilder-level effects. `task_8` is the clearest "
        "high-risk task across keybuilders, especially for `max_pool_w3_d5`.",
        "",
        "## Overall Distribution",
        "",
        "![Colored deviate-score distributions](plots/step2_score_distribution_colored.png)",
        "",
    ]

    lines.extend(
        _markdown_table(
            [
                "config",
                "episodes",
                "cycles",
                "mean",
                "median",
                "p90",
                "p95",
                "p99",
                "max",
                ">=5 / ep mean",
                "zero- >=5 episodes",
            ],
            [
                [
                    f"`{row['config']}`",
                    str(row["episodes"]),
                    str(row["cycles"]),
                    f"{row['mean']:.2f}",
                    f"{row['median']:.2f}",
                    f"{row['p90']:.2f}",
                    f"{row['p95']:.2f}",
                    f"{row['p99']:.2f}",
                    f"{row['max']:.2f}",
                    f"{row['ge5_ep_mean']:.2f}",
                    str(row["zero_ge5_episodes"]),
                ]
                for row in overall_rows
            ],
        )
    )
    lines.extend(["", "Band counts by cycle:", ""])
    lines.extend(
        _markdown_table(
            ["config", "<1", "1-2", "2-5", ">=5"],
            [
                [
                    f"`{row['config']}`",
                    f"{row['<1']} ({row['<1_pct']:.1f}%)",
                    f"{row['1-2']} ({row['1-2_pct']:.1f}%)",
                    f"{row['2-5']} ({row['2-5_pct']:.1f}%)",
                    f"{row['>=5']} ({row['>=5_pct']:.1f}%)",
                ]
                for row in band_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "Interpretation: most cycles are not tiny perturbations. Roughly 44-46% of cycles are "
            "at least `2x` background variation, and about 6-7% are `>=5x`. This is why Step 3 "
            "should focus on top-k high-score cycles rather than random cycle sampling alone.",
            "",
            "## Episode-Level Very-Large Point Counts",
            "",
            "![Per-episode >=5 count histogram](plots/step2_episode_ge5_count_hist.png)",
            "",
            "Most episodes contain a small number of very-large points: usually 0-2, with a long "
            "tail. This supports using `k-grid` values such as `1 3 5`: `k=1` targets the dominant "
            "spike, while `k=3/5` captures the minority of episodes with several high-risk points.",
            "",
            "## Task-Level Risk",
            "",
            "![Task/keybuilder >=5 heatmap](plots/step2_task_keybuilder_ge5_heatmap.png)",
            "",
            "![Pooled task risk ranking](plots/step2_pooled_task_risk_ranking.png)",
            "",
        ]
    )
    english_task_descriptions = {
        "盘子和小模子之间 -> 放到盘子": "between plate and ramekin -> plate",
        "小模子旁边 -> 放到盘子": "next to ramekin -> plate",
        "桌面中央 -> 放到盘子": "table center -> plate",
        "饼干盒上 -> 放到盘子": "on cookie box -> plate",
        "木柜顶层抽屉里 -> 放到盘子": "top drawer cabinet -> plate",
        "小模子上 -> 放到盘子": "on ramekin -> plate",
        "饼干盒旁边 -> 放到盘子": "next to cookie box -> plate",
        "炉灶上 -> 放到盘子": "on stove -> plate",
        "盘子旁边 -> 放到盘子": "next to plate -> plate",
        "木柜上 -> 放到盘子": "on wooden cabinet -> plate",
    }
    lines.extend(
        _markdown_table(
            ["task", "description", "pooled >=5 % cycles", "pooled >=5 / ep mean", "pooled mean score"],
            [
                [
                    str(row["task_id"]),
                    english_task_descriptions.get(row["description"], row["description"]),
                    f"{row['ge5_pct']:.1f}%",
                    f"{row['ge5_ep_mean']:.2f}",
                    f"{row['mean']:.2f}",
                ]
                for row in task_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "Key task finding: `task_8` (`next to plate -> plate`) is the strongest outlier. It has "
            "the highest pooled `>=5` density and dominates the extreme episode list. `task_0` is "
            "second. `task_2` is consistently lower risk.",
            "",
            "## Where High Scores Occur Within Episodes",
            "",
            "![Relative position of >=5 points](plots/step2_ge5_relative_position.png)",
            "",
            "The `>=5` points concentrate in the middle 20-60% and the final 80-100% of episodes. "
            "The early 0-20% region is comparatively quiet. This suggests the score is picking up "
            "interaction phases such as contact/lift/transport and final placement/alignment rather "
            "than initial approach alone.",
            "",
            "## Extreme Episodes",
            "",
            "![Top episodes by >=5 count](plots/step2_top_episodes_ge5.png)",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["rank", "config", "episode", "cycles", "# >=5", "max score", "indices >=5"],
            [
                [
                    str(rank),
                    f"`{row['config']}`",
                    f"`{row['episode']}`",
                    str(row["cycles"]),
                    str(row["ge5_count"]),
                    f"{row['max']:.2f}",
                    f"`{row['indices']}`",
                ]
                for rank, row in enumerate(top_rows, start=1)
            ],
        )
    )
    lines.extend(
        [
            "",
            "These episodes are good qualitative case-study candidates for Step 3 because they "
            "contain sustained or repeated large deviations rather than a single isolated spike.",
            "",
            "## Implications for Step 3",
            "",
            "- Use all three keybuilders in Step 3; their score distributions are close globally, "
            "but task-specific differences remain visible.",
            "- Keep `k-grid 1 3 5`: most episodes have 1-2 very-large points, while the tail episodes "
            "need 3-5 points to cover repeated deviations.",
            "- `task_8`, `task_0`, and high-count episodes in the top-episode table should be "
            "prioritized for visual inspection and debugging.",
            "- Position-wise, include `n-grid` values that test both mid-trajectory and late placement "
            "failures. The late peak means close-to-end recovery remains important.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def _write_report_zh(
    out_path: Path,
    score_dir: Path,
    configs: list[str],
    overall_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# Step 2 偏离分数分析",
        "",
        f"数据来源：`{score_dir}/deviate_score_{{cfg}}.json`。",
        "",
        "完整分析和主要图像由下面这个脚本生成：",
        "",
        "```bash",
        "uv run python -m exp.trajectory_deviation.analysis.analyze_step2_deviate_scores \\",
        f"    --score-dir {score_dir} \\",
        "    --gt-dir exp/trajectory_deviation/data/gt",
        "```",
        "",
        "只画连续分布图和简版 summary 的轻量脚本是：",
        "",
        "```bash",
        "uv run python -m exp.trajectory_deviation.analysis.plot_deviate_score_distribution \\",
        f"    --score-dir {score_dir}",
        "```",
        "",
        "这份分析参考 Step 1a report 的结构，但数据只使用 Step 2 刚得到的偏离分数输出。"
        "每个 replan cycle 的分数定义为 `cache_l2 / max(background_l2, 0.1)`。分数越高，表示 cache replay 的动作相对 GT 的偏离，"
        "比纯 inference 自身背景波动更大。",
        "",
        "## 核心结论",
        "",
        "- 三个 keybuilder 的整体分布非常接近：中位数大约在 `1.82-1.86`，`>=5` 的大偏离点占全部 cycle 的 `6.2-6.7%`。",
        "- 典型 episode 大约有 `21-22` 个 replan cycle，其中大约 `1.3-1.5` 个 cycle 的分数 `>=5`。按中位数看，一个 episode 通常刚好有 `1` 个大偏离点。",
        "- 高风险点不是均匀分布在轨迹里：它们最常出现在轨迹中段，并且在轨迹末段再次出现。每个 episode 最开始的 20% 区间大偏离点最少。",
        "- task 维度的差异比 keybuilder 维度更明显。`task_8` 是最清楚的高风险任务，三个 keybuilder 都高，尤其是 `max_pool_w3_d5`。",
        "",
        "## 整体分布",
        "",
        "![偏离分数连续分布，按阈值着色](plots/step2_score_distribution_colored.png)",
        "",
    ]

    lines.extend(
        _markdown_table(
            [
                "配置",
                "episode 数",
                "cycle 数",
                "均值",
                "中位数",
                "p90",
                "p95",
                "p99",
                "最大值",
                "每个 episode 平均 >=5 数",
                "没有 >=5 的 episode 数",
            ],
            [
                [
                    f"`{row['config']}`",
                    str(row["episodes"]),
                    str(row["cycles"]),
                    f"{row['mean']:.2f}",
                    f"{row['median']:.2f}",
                    f"{row['p90']:.2f}",
                    f"{row['p95']:.2f}",
                    f"{row['p99']:.2f}",
                    f"{row['max']:.2f}",
                    f"{row['ge5_ep_mean']:.2f}",
                    str(row["zero_ge5_episodes"]),
                ]
                for row in overall_rows
            ],
        )
    )
    lines.extend(["", "按 cycle 统计的分数区间：", ""])
    lines.extend(
        _markdown_table(
            ["配置", "<1", "1-2", "2-5", ">=5"],
            [
                [
                    f"`{row['config']}`",
                    f"{row['<1']} ({row['<1_pct']:.1f}%)",
                    f"{row['1-2']} ({row['1-2_pct']:.1f}%)",
                    f"{row['2-5']} ({row['2-5_pct']:.1f}%)",
                    f"{row['>=5']} ({row['>=5_pct']:.1f}%)",
                ]
                for row in band_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "解释：大部分 cycle 不是非常小的扰动。大约 44-46% 的 cycle 至少达到背景波动的 `2x`，约 6-7% 达到 `>=5x`。"
            "所以 Step 3 不应该只随机抽 cycle，而应该重点覆盖 top-k 高分 cycle。",
            "",
            "## Episode 内部大偏离点数量",
            "",
            "![每个 episode 中 >=5 点数量的直方图](plots/step2_episode_ge5_count_hist.png)",
            "",
            "大多数 episode 只包含少量大偏离点，通常是 0-2 个，但存在长尾。"
            "这支持继续使用类似 `k-grid 1 3 5` 的设置：`k=1` 捕捉最主要的尖峰，`k=3/5` 覆盖少数包含多个高风险点的 episode。",
            "",
            "## Task 维度风险",
            "",
            "![task/keybuilder 的 >=5 比例热力图](plots/step2_task_keybuilder_ge5_heatmap.png)",
            "",
            "![合并三个 keybuilder 后的 task 风险排序](plots/step2_pooled_task_risk_ranking.png)",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["task", "任务描述", "合并后 >=5 cycle 占比", "合并后每 episode 平均 >=5 数", "合并后平均分"],
            [
                [
                    str(row["task_id"]),
                    row["description"],
                    f"{row['ge5_pct']:.1f}%",
                    f"{row['ge5_ep_mean']:.2f}",
                    f"{row['mean']:.2f}",
                ]
                for row in task_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "关键 task 结论：`task_8`（盘子旁边 -> 放到盘子）是最强的异常点。"
            "它的合并 `>=5` 密度最高，并且在极端 episode 列表中占比很高。`task_0` 排第二，`task_2` 则稳定地更低风险。",
            "",
            "## 高分点在 Episode 内的位置",
            "",
            "![>=5 点在 episode 内的相对位置](plots/step2_ge5_relative_position.png)",
            "",
            "`>=5` 的点主要集中在 episode 的 20-60% 中段，并且在 80-100% 末段再次增多。"
            "0-20% 的早期阶段相对安静。这说明偏离分数捕捉到的主要不是初始靠近阶段，而更可能是接触、抓取、搬运、最终放置和对齐这些交互阶段。",
            "",
            "## 极端 Episode",
            "",
            "![按 >=5 点数量排序的极端 episode](plots/step2_top_episodes_ge5.png)",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["排名", "配置", "episode", "cycle 数", ">=5 数量", "最大分数", ">=5 的 cycle 索引"],
            [
                [
                    str(rank),
                    f"`{row['config']}`",
                    f"`{row['episode']}`",
                    str(row["cycles"]),
                    str(row["ge5_count"]),
                    f"{row['max']:.2f}",
                    f"`{row['indices']}`",
                ]
                for rank, row in enumerate(top_rows, start=1)
            ],
        )
    )
    lines.extend(
        [
            "",
            "这些 episode 很适合作为 Step 3 的定性 case study，因为它们不是只有一个孤立尖峰，而是包含持续或重复出现的大偏离。",
            "",
            "## 对 Step 3 的影响",
            "",
            "- Step 3 仍然应该包含三个 keybuilder；它们整体分布接近，但 task 级别差异仍然明显。",
            "- 保留 `k-grid 1 3 5`：大多数 episode 只有 1-2 个大偏离点，但长尾 episode 需要 3-5 个点才能覆盖重复偏离。",
            "- `task_8`、`task_0`，以及极端 episode 表里的高计数 episode，应该优先做可视化检查和问题定位。",
            "- 位置选择上，`n-grid` 应该覆盖轨迹中段和末段的失败。末段峰值说明接近结束时的恢复能力仍然很重要。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-dir",
        default="exp/trajectory_deviation/data/deviate_scores",
        help="Directory containing deviate_score_{cfg}.json files.",
    )
    parser.add_argument(
        "--gt-dir",
        default="exp/trajectory_deviation/data/gt",
        help="GT directory used only to read task descriptions from h5 attrs.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Config ids to analyze.",
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="Output Markdown path. Defaults to SCORE_DIR/step2_deviate_score_analysis.md.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Output plot directory. Defaults to SCORE_DIR/plots.",
    )
    parser.add_argument(
        "--language",
        choices=["en", "zh"],
        default="en",
        help="Report language. Defaults to en to preserve the original report path.",
    )
    parser.add_argument("--top-episodes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_dir = Path(args.score_dir)
    gt_dir = Path(args.gt_dir)
    out_md = Path(args.out_md) if args.out_md else score_dir / "step2_deviate_score_analysis.md"
    plots_dir = Path(args.plots_dir) if args.plots_dir else score_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = _load_episode_records(score_dir, args.configs)
    task_ids = sorted({record.task_id for record in records})
    task_descriptions = _read_task_descriptions(gt_dir, task_ids)

    overall_rows = _overall_rows(records, args.configs)
    band_rows = _band_rows(records, args.configs)
    task_rows = _task_rows(records, task_descriptions)
    top_rows = _top_episode_rows(records, args.top_episodes)

    _plot_distribution(records, args.configs, plots_dir / "step2_score_distribution_colored.png")
    _plot_task_heatmap(records, args.configs, plots_dir / "step2_task_keybuilder_ge5_heatmap.png")
    _plot_episode_hist(records, args.configs, plots_dir / "step2_episode_ge5_count_hist.png")
    _plot_relative_positions(records, args.configs, plots_dir / "step2_ge5_relative_position.png")
    _plot_task_ranking(task_rows, plots_dir / "step2_pooled_task_risk_ranking.png")
    _plot_top_episodes(top_rows, plots_dir / "step2_top_episodes_ge5.png")
    write_report = _write_report_zh if args.language == "zh" else _write_report_en
    write_report(out_md, score_dir, args.configs, overall_rows, band_rows, task_rows, top_rows)

    print(f"Wrote {out_md}")
    print(f"Wrote {plots_dir}")


if __name__ == "__main__":
    main()
