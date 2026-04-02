from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_common import OUTPUTS_DIR, load_results

PLOTS_DIR = OUTPUTS_DIR / "plots"
MODEL_ORDER = ["S", "B", "L", "XL"]
MODEL_WEIGHTS = {
    "S": 1,
    "B": 4,
    "L": 8,
    "XL": 16,
}

BASELINE_KEY_RE = re.compile(r"baseline_model_(?P<model>\w+)_num_steps_(?P<num_steps>\d+)")
SPEC_KEY_RE = re.compile(
    r"speculative_draft_(?P<draft>\w+)_base_(?P<base>\w+)"
    r"_steps_(?P<num_steps>\d+)_threshold_(?P<threshold>[0-9.]+)"
    r"_draft_k_(?P<spec_k>\d+)_(?P<mode>overlap|sequential)"
)
TWO_PICARD_KEY_RE = re.compile(
    r"two_picard_draft_(?P<draft>\w+)_base_(?P<base>\w+)"
    r"_steps_(?P<num_steps>\d+)_threshold_(?P<threshold>[0-9.]+)"
    r"_draft_init_(?P<draft_init>\d+)"
)
GRID_KEY_RE = re.compile(
    r"two_picard_grid_draft_(?P<draft>\w+)_base_(?P<base>\w+)"
    r"_steps_(?P<num_steps>\d+)_class_(?P<class_idx>\d+)"
)


@dataclass(frozen=True)
class BaselineConfig:
    model: str
    num_steps: int
    threshold: float


@dataclass(frozen=True)
class SpeculativeConfig:
    draft: str
    base: str
    num_steps: int
    threshold: float
    spec_k: int
    overlap: bool


@dataclass(frozen=True)
class TwoPicardConfig:
    draft: str
    base: str
    num_steps: int
    threshold: float
    draft_init: int


@dataclass(frozen=True)
class GridPoint:
    draft: str
    base: str
    num_steps: int
    draft_iters: int
    base_iters: int
    kid: float

    @property
    def pair_label(self) -> str:
        return f"{self.draft} -> {self.base}"

    @property
    def config_label(self) -> str:
        return f"{self.draft} -> {self.base} ({self.draft_iters}, {self.base_iters})"

    @property
    def weighted_iters(self) -> float:
        return (
            MODEL_WEIGHTS.get(self.draft, 1) * self.draft_iters
            + MODEL_WEIGHTS.get(self.base, 1) * self.base_iters
        )


@dataclass(frozen=True)
class MSSIMPoint:
    draft: str
    base: str
    num_steps: int
    draft_iters: int
    base_iters: int
    mssim_score: float

    @property
    def pair_label(self) -> str:
        return f"{self.draft} -> {self.base}"

    @property
    def weighted_iters(self) -> float:
        return (
            MODEL_WEIGHTS.get(self.draft, 1) * self.draft_iters
            + MODEL_WEIGHTS.get(self.base, 1) * self.base_iters
        )


def sort_models(models: Iterable[str]) -> list[str]:
    return sorted(models, key=lambda model: MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER))


def metric_mean_std(records: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = np.array([record[field] for record in records if record.get(field) is not None], dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def parse_baseline_results(raw: dict[str, list[dict[str, Any]]]) -> dict[BaselineConfig, list[dict[str, Any]]]:
    parsed: dict[BaselineConfig, list[dict[str, Any]]] = {}
    for key, records in raw.items():
        if not records:
            continue
        match = BASELINE_KEY_RE.fullmatch(key)
        if not match:
            continue
        model = records[0].get("model", match.group("model"))
        num_steps = int(records[0].get("num_steps", match.group("num_steps")))
        by_threshold: dict[float, list[dict[str, Any]]] = {}
        for record in records:
            threshold = float(record["threshold"])
            by_threshold.setdefault(threshold, []).append(record)
        for threshold, threshold_records in by_threshold.items():
            parsed[BaselineConfig(model=model, num_steps=num_steps, threshold=threshold)] = threshold_records
    return parsed


def parse_speculative_results(raw: dict[str, list[dict[str, Any]]]) -> dict[SpeculativeConfig, list[dict[str, Any]]]:
    parsed: dict[SpeculativeConfig, list[dict[str, Any]]] = {}
    for key, records in raw.items():
        match = SPEC_KEY_RE.fullmatch(key)
        if not match:
            continue
        sample = records[0] if records else {}
        cfg = SpeculativeConfig(
            draft=sample.get("draft_model", match.group("draft")),
            base=sample.get("base_model", match.group("base")),
            num_steps=int(sample.get("num_steps", match.group("num_steps"))),
            threshold=float(sample.get("threshold", match.group("threshold"))),
            spec_k=int(sample.get("spec_k", match.group("spec_k"))),
            overlap=bool(sample.get("overlap", match.group("mode") == "overlap")),
        )
        parsed[cfg] = records
    return parsed


def parse_two_picard_results(raw: dict[str, list[dict[str, Any]]]) -> dict[TwoPicardConfig, list[dict[str, Any]]]:
    parsed: dict[TwoPicardConfig, list[dict[str, Any]]] = {}
    for key, records in raw.items():
        match = TWO_PICARD_KEY_RE.fullmatch(key)
        if not match:
            continue
        sample = records[0] if records else {}
        cfg = TwoPicardConfig(
            draft=sample.get("draft_model", match.group("draft")),
            base=sample.get("base_model", match.group("base")),
            num_steps=int(sample.get("num_steps", match.group("num_steps"))),
            threshold=float(sample.get("threshold", match.group("threshold"))),
            draft_init=int(sample.get("draft_init", match.group("draft_init"))),
        )
        parsed[cfg] = records
    return parsed


def parse_grid_results(raw: dict[str, list[dict[str, Any]]]) -> list[GridPoint]:
    grouped: dict[tuple[str, str, int, int, int], list[float]] = {}
    for key, records in raw.items():
        match = GRID_KEY_RE.fullmatch(key)
        if not match:
            continue
        for record in records:
            draft = record.get("draft_model", match.group("draft"))
            base = record.get("base_model", match.group("base"))
            num_steps = int(record.get("num_steps", match.group("num_steps")))
            point_key = (draft, base, num_steps, int(record["draft_iters"]), int(record["base_iters"]))
            grouped.setdefault(point_key, []).append(float(record["kid"]))

    return [
        GridPoint(
            draft=draft,
            base=base,
            num_steps=num_steps,
            draft_iters=draft_iters,
            base_iters=base_iters,
            kid=float(np.mean(kids)),
        )
        for (draft, base, num_steps, draft_iters, base_iters), kids in grouped.items()
    ]


def parse_mssim_results(raw: dict[str, list[dict[str, Any]]]) -> list[MSSIMPoint]:
    grouped: dict[tuple[str, str, int, int, int], list[float]] = {}
    for key, records in raw.items():
        match = GRID_KEY_RE.fullmatch(key)
        if not match:
            continue
        for record in records:
            draft = record.get("draft_model", match.group("draft"))
            base = record.get("base_model", match.group("base"))
            num_steps = int(record.get("num_steps", match.group("num_steps")))
            point_key = (draft, base, num_steps, int(record["draft_iters"]), int(record["base_iters"]))
            grouped.setdefault(point_key, []).append(float(record["mssim_score"]))

    return [
        MSSIMPoint(
            draft=draft,
            base=base,
            num_steps=num_steps,
            draft_iters=draft_iters,
            base_iters=base_iters,
            mssim_score=float(np.mean(scores)),
        )
        for (draft, base, num_steps, draft_iters, base_iters), scores in grouped.items()
    ]


def ensure_plot_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    ensure_plot_dir()
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_two_picard_wallclock(
    baseline: dict[BaselineConfig, list[dict[str, Any]]],
    two_picard: dict[TwoPicardConfig, list[dict[str, Any]]],
) -> None:
    thresholds = sorted({cfg.threshold for cfg in two_picard})
    pairs = sorted({(cfg.draft, cfg.base) for cfg in two_picard}, key=lambda pair: tuple(sort_models(pair)))
    steps = sorted({cfg.num_steps for cfg in two_picard})

    for threshold in thresholds:
        fig, axes = plt.subplots(
            len(steps),
            len(pairs),
            figsize=(5 * max(len(pairs), 1), 4 * max(len(steps), 1)),
            squeeze=False,
        )
        fig.suptitle(f"Two-Picard wall clock vs baseline (threshold={threshold})", fontsize=13, fontweight="bold")

        for row, num_steps in enumerate(steps):
            for col, (draft, base) in enumerate(pairs):
                ax = axes[row][col]
                baseline_cfg = BaselineConfig(model=base, num_steps=num_steps, threshold=threshold)
                baseline_records = baseline.get(baseline_cfg)
                if not baseline_records:
                    print(
                        "Skipping two-picard panel:"
                        f" missing baseline for {base}, steps={num_steps}, threshold={threshold}"
                    )
                    ax.set_visible(False)
                    continue

                base_by_idx = {record["img_idx"]: record["wall_clock_s"] for record in baseline_records}
                configs = sorted(
                    [cfg for cfg in two_picard if cfg.draft == draft and cfg.base == base and cfg.num_steps == num_steps and cfg.threshold == threshold],
                    key=lambda cfg: cfg.draft_init,
                )
                box_data: list[list[float]] = []
                labels: list[str] = []
                medians: list[float] = []

                for cfg in configs:
                    ratios = [
                        100.0 * record["wall_clock_s"] / base_by_idx[record["img_idx"]]
                        for record in two_picard[cfg]
                        if record["img_idx"] in base_by_idx and base_by_idx[record["img_idx"]] > 0
                    ]
                    if not ratios:
                        continue
                    box_data.append(ratios)
                    labels.append(str(cfg.draft_init))
                    medians.append(float(np.median(ratios)))

                if not box_data:
                    ax.set_visible(False)
                    continue

                boxplot = ax.boxplot(
                    box_data,
                    patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.4},
                )
                for patch in boxplot["boxes"]:
                    patch.set_facecolor("#4C78A8")
                    patch.set_alpha(0.65)

                ax.plot(range(1, len(medians) + 1), medians, color="#1F3552", linewidth=1.2, marker="o", markersize=3)
                ax.axhline(100, color="#E45756", linestyle="--", linewidth=1.0)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_title(f"{draft} -> {base}", fontsize=10)
                ax.set_xlabel("Draft init iters", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"steps={num_steps}\n% of baseline wall-clock", fontsize=9)

        save_figure(fig, f"02_twopic_wallclock_threshold_{threshold}.png")


def acceptance_series(record: dict[str, Any]) -> list[float]:
    if "acceptance_history" in record:
        return [float(value) for value in record["acceptance_history"]]
    history = record.get("best_draft_indices_history", [])
    return [float(any(candidate_idx > 0 for candidate_idx in step)) for step in history]


def plot_speculative_acceptance(speculative: dict[SpeculativeConfig, list[dict[str, Any]]]) -> None:
    thresholds = sorted({cfg.threshold for cfg in speculative})
    spec_ks = sorted({cfg.spec_k for cfg in speculative})
    pairs = sorted({(cfg.draft, cfg.base) for cfg in speculative}, key=lambda pair: tuple(sort_models(pair)))
    steps = sorted({cfg.num_steps for cfg in speculative})

    for threshold in thresholds:
        for spec_k in spec_ks:
            fig, axes = plt.subplots(2, len(pairs), figsize=(5 * max(len(pairs), 1), 8), squeeze=False)
            fig.suptitle(f"Speculative acceptance (threshold={threshold}, K={spec_k})", fontsize=13, fontweight="bold")

            for col, (draft, base) in enumerate(pairs):
                for row, overlap in enumerate([True, False]):
                    ax = axes[row][col]
                    plotted = False

                    for line_style, num_steps in zip(["-", "--", ":", "-."], steps):
                        cfg = SpeculativeConfig(
                            draft=draft,
                            base=base,
                            num_steps=num_steps,
                            threshold=threshold,
                            spec_k=spec_k,
                            overlap=overlap,
                        )
                        records = speculative.get(cfg)
                        if not records:
                            continue

                        series = [acceptance_series(record) for record in records]
                        max_len = max(len(item) for item in series)
                        padded = np.full((len(series), max_len), np.nan)
                        for idx, item in enumerate(series):
                            padded[idx, : len(item)] = np.array(item, dtype=float)

                        means = np.nanmean(padded, axis=0)
                        stds = np.nanstd(padded, axis=0)
                        xs = np.arange(1, max_len + 1)
                        label = f"steps={num_steps} (mean={float(np.nanmean(means)):.2f})"
                        color = "#4C78A8" if overlap else "#F58518"
                        ax.plot(xs, means, linestyle=line_style, color=color, linewidth=1.8, label=label)
                        ax.fill_between(xs, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color=color, alpha=0.12)
                        plotted = True

                    ax.set_ylim(0.0, 1.05)
                    ax.set_title(f"{draft} -> {base} [{'overlap' if overlap else 'sequential'}]", fontsize=10)
                    ax.set_xlabel("Outer iteration", fontsize=9)
                    if col == 0:
                        ax.set_ylabel("Acceptance rate", fontsize=9)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    if plotted:
                        ax.legend(fontsize=7)

            save_figure(fig, f"03_spec_acceptance_threshold_{threshold}_K{spec_k}.png")


def pareto_frontier(points: list[GridPoint]) -> list[GridPoint]:
    ordered = sorted(points, key=lambda point: (point.weighted_iters, point.kid))
    frontier: list[GridPoint] = []
    best_kid = math.inf
    for point in ordered:
        if point.kid < best_kid:
            frontier.append(point)
            best_kid = point.kid
    return frontier


def pareto_frontier_mssim(points: list[MSSIMPoint]) -> list[MSSIMPoint]:
    ordered = sorted(points, key=lambda point: (point.weighted_iters, -point.mssim_score))
    frontier: list[MSSIMPoint] = []
    best_mssim = -math.inf
    for point in ordered:
        if point.mssim_score > best_mssim:
            frontier.append(point)
            best_mssim = point.mssim_score
    return frontier


def plot_weighted_iters_vs_kid(grid_points: list[GridPoint]) -> None:
    if not grid_points:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    pair_labels = sorted({point.pair_label for point in grid_points})
    colors = {
        label: color
        for label, color in zip(
            pair_labels,
            ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"],
        )
    }

    for pair_label in pair_labels:
        pair_points = [point for point in grid_points if point.pair_label == pair_label]
        xs = [point.weighted_iters for point in pair_points]
        ys = [point.kid for point in pair_points]
        ax.scatter(xs, ys, s=70, alpha=0.85, color=colors[pair_label], label=pair_label, edgecolors="black", linewidths=0.4)

        for point in pair_points:
            ax.annotate(
                f"({point.draft_iters}, {point.base_iters})",
                (point.weighted_iters, point.kid),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    frontier = pareto_frontier(grid_points)
    ax.plot(
        [point.weighted_iters for point in frontier],
        [point.kid for point in frontier],
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Pareto frontier",
    )

    ax.set_title("Weighted sequential iterations vs KID", fontsize=13, fontweight="bold")
    ax.set_xlabel("Weighted sequential iterations", fontsize=10)
    ax.set_ylabel("KID", fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8)

    weight_caption = ", ".join(f"{model}={weight}" for model, weight in MODEL_WEIGHTS.items())
    ax.text(
        0.01,
        0.01,
        f"Weights: {weight_caption}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    save_figure(fig, "04_weighted_iters_vs_kid.png")


def plot_weighted_iters_vs_mssim(points: list[MSSIMPoint]) -> None:
    if not points:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    pair_labels = sorted({point.pair_label for point in points})
    colors = {
        label: color
        for label, color in zip(
            pair_labels,
            ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"],
        )
    }

    for pair_label in pair_labels:
        pair_points = [point for point in points if point.pair_label == pair_label]
        xs = [point.weighted_iters for point in pair_points]
        ys = [point.mssim_score for point in pair_points]
        ax.scatter(xs, ys, s=70, alpha=0.85, color=colors[pair_label], label=pair_label, edgecolors="black", linewidths=0.4)

        for point in pair_points:
            ax.annotate(
                f"({point.draft_iters}, {point.base_iters})",
                (point.weighted_iters, point.mssim_score),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    frontier = pareto_frontier_mssim(points)
    ax.plot(
        [point.weighted_iters for point in frontier],
        [point.mssim_score for point in frontier],
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Pareto frontier",
    )

    ax.set_title("Weighted sequential iterations vs MSSIM", fontsize=13, fontweight="bold")
    ax.set_xlabel("Weighted sequential iterations", fontsize=10)
    ax.set_ylabel("MSSIM", fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=8)

    weight_caption = ", ".join(f"{model}={weight}" for model, weight in MODEL_WEIGHTS.items())
    ax.text(
        0.01,
        0.01,
        f"Weights: {weight_caption}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    save_figure(fig, "05_weighted_iters_vs_mssim.png")


def main() -> None:
    baseline = parse_baseline_results(load_results("baseline"))
    mssim_points = parse_mssim_results(load_results("mssim"))
    speculative = parse_speculative_results(load_results("speculative"))
    two_picard = parse_two_picard_results(load_results("two_picard_time"))
    grid_points = parse_grid_results(load_results("two_picard_grid"))

    if baseline and two_picard:
        plot_two_picard_wallclock(baseline, two_picard)
    else:
        print("Skipping two-picard wall-clock plot: missing baseline or two_picard_time results.")

    if speculative:
        plot_speculative_acceptance(speculative)
    else:
        print("Skipping speculative acceptance plot: missing speculative results.")

    if grid_points:
        plot_weighted_iters_vs_kid(grid_points)
    else:
        print("Skipping weighted-iterations vs KID plot: missing two_picard_grid results.")

    if mssim_points:
        plot_weighted_iters_vs_mssim(mssim_points)
    else:
        print("Skipping weighted-iterations vs MSSIM plot: missing mssim results.")


if __name__ == "__main__":
    main()
