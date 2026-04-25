from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
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
from inference import model_call_cfg

SPEC_NAME = "euler"
NUM_STEPS_SWEEP = [8, 16, 32, 128, 256]


@dataclass
class EulerStat:
    img_idx: int
    model: str
    num_steps: int
    wall_clock_s: float


def euler_iterate(
    model: torch.nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor,
    y_null: torch.Tensor,
    num_steps: int,
    cfg_scale: float,
) -> torch.Tensor:
    dt = 1.0 / num_steps
    x = x0
    batch_size = x.shape[0]

    for step in range(num_steps):
        t = torch.full(
            (batch_size,),
            step / num_steps,
            device=x.device,
            dtype=x.dtype,
        )
        v = model_call_cfg(model, x, t, y, y_null, cfg_scale)
        x = x + v * dt

    return x


def run(num_images: int = NUM_IMAGES, force: bool = False) -> None:
    x, y, y_null = make_eval_batch(num_images)
    vae = get_vae()

    for model_name in tqdm(get_available_models(), desc="euler models"):
        model = load_model(model_name)

        for num_steps in NUM_STEPS_SWEEP:
            eval_key = f"euler_model_{model_name}_num_steps_{num_steps}"
            if not force and result_exists(SPEC_NAME, eval_key) and images_complete(SPEC_NAME, eval_key, num_images):
                continue

            records = []
            for idx in tqdm(range(num_images), desc=eval_key, leave=False):
                z = x[idx]
                labels = y[idx]
                null_labels = y_null[idx]

                t0 = time.perf_counter()
                output = euler_iterate(
                    model,
                    z,
                    labels,
                    null_labels,
                    num_steps,
                    CFG_SCALE,
                )

                decoded_image = vae.decode(output / 0.18215).sample
                save_decoded_image(SPEC_NAME, eval_key, idx, decoded_image)

                records.append(
                    asdict(
                        EulerStat(
                            img_idx=idx,
                            model=model_name,
                            num_steps=num_steps,
                            wall_clock_s=time.perf_counter() - t0,
                        )
                    )
                )

            save_result(SPEC_NAME, eval_key, records)


if __name__ == "__main__":
    with torch.no_grad():
        run()
