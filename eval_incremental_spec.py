from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from eval_common import (
    CFG_SCALE,
    OUTPUTS_DIR,
    get_available_models,
    load_model,
    load_results,
    make_eval_batch,
    get_vae,
    images_complete,
    save_decoded_image,
    save_results,
)
from inference import (
    piecewise_picard_trajectory,
    upscaling_piecewise_picard,
)

NUM_IMAGES = 32

SPEC_NAME = "incremental_spec"
PLOTS_DIR = OUTPUTS_DIR / "plots"
FINAL_NUM_STEPS = 256 + 1
STEP_SIZE = 33
THRESHOLDS = [0.05, 0.1]
CONFIGS = [
    {"label": "direct_257", "num_steps_init": FINAL_NUM_STEPS, "multiples": []},
    {"label": "upscale_x8", "num_steps_init": STEP_SIZE, "multiples": [8]},
    # {"label": "upscale_x2_x4", "num_steps_init": STEP_SIZE, "multiples": [2, 4]},
    # {"label": "upscale_x4_x2", "num_steps_init": STEP_SIZE, "multiples": [4, 2]},
    # {"label": "upscale_x2_x2_x2", "num_steps_init": STEP_SIZE, "multiples": [2, 2, 2]},
]

MODELS = ["B", "L"]

@dataclass
class IncrementalSpecStat:
    img_idx: int
    model: str
    threshold: float
    final_num_steps: int
    step_size: int
    config_label: str
    num_steps_init: int
    multiples: list[int]
    total_iters: int
    stage_num_steps: list[int]
    stage_iters: list[int]
    stage_multiples: list[int]
    stage_residual_histories: list[list[list[float]]]


def _stage_multiples(num_steps_init: int, stage_num_steps: list[int]) -> list[int]:
    multiples: list[int] = []
    prev_num_steps = num_steps_init
    for num_steps in stage_num_steps[1:]:
        if prev_num_steps <= 1:
            multiples.append(1)
        else:
            multiples.append(int((num_steps - 1) // (prev_num_steps - 1)))
        prev_num_steps = int(num_steps)
    return multiples


def ensure_plot_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_store() -> dict[str, list[dict]]:
    raw = load_results(SPEC_NAME)
    return raw if isinstance(raw, dict) else {}


def _save_store(store: dict[str, list[dict]]) -> None:
    save_results(SPEC_NAME, store)


def _eval_key(model_name: str, threshold: float) -> str:
    return (
        f"incremental_spec_model_{model_name}"
        f"_steps_{FINAL_NUM_STEPS}"
        f"_step_size_{STEP_SIZE}"
        f"_threshold_{threshold}"
    )


def _validate_config(config: dict[str, object]) -> None:
    num_steps = int(config["num_steps_init"])
    for multiple in config["multiples"]:  # type: ignore[index]
        num_steps = (num_steps - 1) * int(multiple) + 1
    if num_steps != FINAL_NUM_STEPS:
        raise ValueError(f"Config {config['label']} does not end at {FINAL_NUM_STEPS} steps.")


def _config_eval_key(eval_key: str, config_label: str) -> str:
    return f"{eval_key}_config_{config_label}"


def _stage_eval_key(eval_key: str, config_label: str, stage_idx: int, num_steps: int) -> str:
    return f"{_config_eval_key(eval_key, config_label)}_stage_{stage_idx:02d}_steps_{num_steps}"


def _config_image_keys(eval_key: str, config: dict[str, object]) -> list[str]:
    config_label = str(config["label"])
    stage_num_steps = [int(config["num_steps_init"])]
    current_num_steps = int(config["num_steps_init"])
    for multiple in config["multiples"]:  # type: ignore[index]
        current_num_steps = (current_num_steps - 1) * int(multiple) + 1
        stage_num_steps.append(current_num_steps)

    return [_config_eval_key(eval_key, config_label)] + [
        _stage_eval_key(eval_key, config_label, stage_idx, num_steps)
        for stage_idx, num_steps in enumerate(stage_num_steps, start=1)
    ]


def _images_complete_for_config(eval_key: str, config: dict[str, object], num_images: int) -> bool:
    return all(images_complete(SPEC_NAME, image_key, num_images) for image_key in _config_image_keys(eval_key, config))


def _save_stage_images(
    vae,
    eval_key: str,
    config_label: str,
    image_idx: int,
    final_output: torch.Tensor,
    stage_results,
) -> None:
    save_decoded_image(
        SPEC_NAME,
        _config_eval_key(eval_key, config_label),
        image_idx,
        vae.decode(final_output / 0.18215).sample,
    )
    for stage_idx, stage in enumerate(stage_results, start=1):
        if stage.final_latent is None:
            continue
        save_decoded_image(
            SPEC_NAME,
            _stage_eval_key(eval_key, config_label, stage_idx, int(stage.num_steps)),
            image_idx,
            vae.decode(stage.final_latent / 0.18215).sample,
        )


def run(num_images: int = NUM_IMAGES, force: bool = False) -> None:
    for config in CONFIGS:
        _validate_config(config)


    x, y, y_null = make_eval_batch(num_images)
    store = _load_store()
    vae = get_vae()

    for model_name in tqdm(MODELS, desc="incremental models"):
        model = load_model(model_name)
        for threshold in THRESHOLDS:
            eval_key = _eval_key(model_name, threshold)
            existing_records = [] if force else list(store.get(eval_key, []))
            done_pairs = {
                (int(record["img_idx"]), str(record["config_label"]))
                for record in existing_records
            }
            records = [] if force else existing_records

            for idx in tqdm(range(num_images), desc=eval_key, leave=False):
                for config in CONFIGS:
                    pair_key = (idx, str(config["label"]))
                    if pair_key in done_pairs and _images_complete_for_config(eval_key, config, num_images):
                        continue
                    if pair_key in done_pairs:
                        records = [
                            record
                            for record in records
                            if (int(record["img_idx"]), str(record["config_label"])) != pair_key
                        ]

                    if config["multiples"]:
                        output, stats = upscaling_piecewise_picard(
                            model=model,
                            x=x[idx],
                            y=y[idx],
                            y_null=y_null[idx],
                            num_steps_init=int(config["num_steps_init"]),
                            multiples=[int(v) for v in config["multiples"]],  # type: ignore[list-item]
                            cfg_scale=CFG_SCALE,
                            threshold=threshold,
                            group_size=STEP_SIZE
                        )
                        total_iters = stats.total_iterations
                        stage_results = stats.stages
                    else:
                        output_traj, stage_result = piecewise_picard_trajectory(
                            model=model,
                            x=x[idx],
                            y=y[idx],
                            y_null=y_null[idx],
                            num_steps=FINAL_NUM_STEPS,
                            cfg_scale=CFG_SCALE,
                            threshold=threshold,
                            group_size=STEP_SIZE,
                        )
                        stage_results = [stage_result]
                        total_iters = stage_result.iterations
                        output = output_traj[-1]

                    _save_stage_images(vae, eval_key, str(config["label"]), idx, output, stage_results)

                    stage_num_steps = [int(stage.num_steps) for stage in stage_results]
                    stage_iters = [int(stage.iterations) for stage in stage_results]
                    stage_residual_histories = [
                        [[float(value) for value in residuals] for residuals in stage.residual_history]
                        for stage in stage_results
                    ]

                    records.append(
                        asdict(
                            IncrementalSpecStat(
                                img_idx=idx,
                                model=model_name,
                                threshold=threshold,
                                final_num_steps=FINAL_NUM_STEPS,
                                step_size=STEP_SIZE,
                                config_label=str(config["label"]),
                                num_steps_init=int(config["num_steps_init"]),
                                multiples=[int(v) for v in config["multiples"]],  # type: ignore[list-item]
                                total_iters=int(total_iters),
                                stage_num_steps=stage_num_steps,
                                stage_iters=stage_iters,
                                stage_multiples=_stage_multiples(int(config["num_steps_init"]), stage_num_steps),
                                stage_residual_histories=stage_residual_histories,
                            )
                        )
                    )
                    done_pairs.add(pair_key)
                    store[eval_key] = records
                    _save_store(store)


def plot() -> None:
    store = _load_store()
    if not store:
        print("No incremental_spec results found.")
        return

    ensure_plot_dir()

    thresholds = sorted(
        {
            float(record["threshold"])
            for records in store.values()
            for record in records
            if "threshold" in record
        }
    )
    model_names = sorted(
        {
            str(record["model"])
            for records in store.values()
            for record in records
            if "model" in record
        }
    )

    for threshold in thresholds:
        fig, axes = plt.subplots(
            len(model_names),
            1,
            figsize=(7, 4 * max(1, len(model_names))),
            squeeze=False,
        )
        fig.suptitle(
            f"Upscaling piecewise Picard comparison (steps={FINAL_NUM_STEPS}, step size={STEP_SIZE}, threshold={threshold})",
            fontsize=13,
            fontweight="bold",
        )

        for row, model_name in enumerate(model_names):
            model_records = [
                record
                for records in store.values()
                for record in records
                if str(record.get("model")) == model_name and float(record.get("threshold")) == threshold
            ]
            grouped: dict[str, list[dict]] = {}
            for record in model_records:
                grouped.setdefault(str(record["config_label"]), []).append(record)

            labels = sorted(grouped.keys())
            if not labels:
                axes[row][0].set_visible(False)
                continue

            iter_data = [[int(record["total_iters"]) for record in grouped[label]] for label in labels]
            iter_means = [float(np.mean(values)) for values in iter_data]
            xs = np.arange(1, len(labels) + 1)

            iter_ax = axes[row][0]
            iter_boxplot = iter_ax.boxplot(
                iter_data,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.4},
            )
            for patch in iter_boxplot["boxes"]:
                patch.set_facecolor("#4C78A8")
                patch.set_alpha(0.65)
            iter_ax.plot(xs, iter_means, color="#1F3552", marker="o", linewidth=1.2)
            iter_ax.set_title(f"{model_name}: total iterations", fontsize=10)
            iter_ax.set_xticks(xs)
            iter_ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            iter_ax.set_ylabel("Iterations", fontsize=9)

        fig.tight_layout()
        output_path = PLOTS_DIR / f"incremental_spec_iters_threshold_{threshold}.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not args.plot_only:
        run(num_images=args.num_images, force=args.force)
    plot()


if __name__ == "__main__":
    with torch.no_grad():
        main()
