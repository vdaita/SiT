from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from tqdm import tqdm

from eval_common import CFG_SCALE, NUM_IMAGES, get_available_models, load_model, make_eval_batch, result_exists, save_result
from inference import two_picard_trajectory

SPEC_NAME = "two_picard_time"
DRAFT_INIT_SWEEP = [1, 2, 4, 8]
MODEL_PAIRS = [
    {"draft": "S", "base": "B", "num_steps": [16, 32], "thresholds": [0.05, 0.1]},
    {"draft": "S", "base": "L", "num_steps": [16, 32], "thresholds": [0.05, 0.1]},
    {"draft": "B", "base": "L", "num_steps": [16, 32], "thresholds": [0.05, 0.1]},
]


@dataclass
class TwoPicardTimingStat:
    img_idx: int
    wall_clock_s: float
    draft_iters: int
    base_iters: int
    draft_residual_history: list[list[float]]
    base_residual_history: list[list[float]]


def run(num_images: int = NUM_IMAGES, force: bool = False) -> None:
    available = set(get_available_models())
    x, y, y_null = make_eval_batch(num_images)

    for pair in tqdm(MODEL_PAIRS, desc="two-picard pairs"):
        if pair["draft"] not in available or pair["base"] not in available:
            continue
        base_model = load_model(pair["base"])
        draft_model = load_model(pair["draft"])

        for num_steps in pair["num_steps"]:
            for threshold in pair["thresholds"]:
                for draft_init in DRAFT_INIT_SWEEP:
                    eval_key = (
                        f"two_picard_draft_{pair['draft']}_base_{pair['base']}"
                        f"_steps_{num_steps}_threshold_{threshold}_draft_init_{draft_init}"
                    )
                    if not force and result_exists(SPEC_NAME, eval_key):
                        continue

                    records = []
                    for idx in tqdm(range(num_images), desc=eval_key, leave=False):
                        t0 = time.perf_counter()
                        _, stats = two_picard_trajectory(
                            base_model,
                            draft_model,
                            x[idx],
                            y[idx],
                            y_null[idx],
                            num_steps,
                            draft_init,
                            CFG_SCALE,
                            threshold,
                        )
                        records.append(
                            asdict(
                                TwoPicardTimingStat(
                                    img_idx=idx,
                                    wall_clock_s=time.perf_counter() - t0,
                                    draft_iters=stats.draft_iters,
                                    base_iters=stats.base_iters,
                                    draft_residual_history=stats.draft_residual_history,
                                    base_residual_history=stats.base_residual_history,
                                )
                            )
                        )

                    save_result(SPEC_NAME, eval_key, records)


if __name__ == "__main__":
    run()
