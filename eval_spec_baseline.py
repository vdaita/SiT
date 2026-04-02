from __future__ import annotations

import torch

from dataclasses import asdict, dataclass
from typing import List

from tqdm import tqdm

from eval_common import (
    CFG_SCALE,
    NUM_IMAGES,
    get_available_models,
    get_vae,
    images_complete,
    load_model,
    make_eval_batch,
    result_exists,
    save_decoded_image,
    save_result,
)
from inference import picard_trajectory

SPEC_NAME = "baseline"
NUM_STEPS_SWEEP = [8, 16, 32, 128]
THRESHOLDS = [0.01, 0.05, 0.1]


@dataclass
class BaselineStat:
    img_idx: int
    model: str
    num_steps: int
    wall_clock_s: float
    iters: int
    residual: List[float]
    threshold: float


def run(num_images: int = NUM_IMAGES, force: bool = False) -> None:
    x, y, y_null = make_eval_batch(num_images)
    vae = get_vae()

    for model_name in tqdm(get_available_models(), desc="baseline models"):
        model = load_model(model_name)
        for num_steps in NUM_STEPS_SWEEP:
            eval_key = f"baseline_model_{model_name}_num_steps_{num_steps}"
            if not force and result_exists(SPEC_NAME, eval_key) and images_complete(SPEC_NAME, eval_key, num_images):
                continue

            records = []
            for idx in tqdm(range(num_images), desc=eval_key, leave=False):
                output, stats = picard_trajectory(
                    model,
                    x[idx],
                    y[idx],
                    y_null[idx],
                    num_steps,
                    CFG_SCALE,
                    THRESHOLDS,
                )
                decoded_image = vae.decode(output / 0.18215).sample
                save_decoded_image(SPEC_NAME, eval_key, idx, decoded_image)
                for threshold_idx, threshold in enumerate(THRESHOLDS):
                    records.append(
                        asdict(
                            BaselineStat(
                                img_idx=idx,
                                model=model_name,
                                num_steps=num_steps,
                                wall_clock_s=stats.durations[threshold_idx],
                                iters=stats.iters[threshold_idx],
                                residual=stats.residual_history[min(stats.iters[threshold_idx] - 1, len(stats.residual_history) - 1)],
                                threshold=threshold,
                            )
                        )
                    )

            save_result(SPEC_NAME, eval_key, records)


if __name__ == "__main__":
    with torch.no_grad():
        run()
