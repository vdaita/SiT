from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from tqdm import tqdm

from eval_common import CFG_SCALE, NUM_IMAGES, get_available_models, load_model, make_eval_batch, result_exists, save_result
from inference import speculative_trajectory

SPEC_NAME = "speculative"
SPECULATIVE_CONFIGS = [
    {"draft": "S", "base": "B", "spec_k": 4, "num_steps": [16, 32], "thresholds": [0.05, 0.1]},
    {"draft": "S", "base": "L", "spec_k": 4, "num_steps": [16, 32], "thresholds": [0.05, 0.1]},
    {"draft": "B", "base": "L", "spec_k": 2, "num_steps": [16, 32], "thresholds": [0.05, 0.1]},
]


@dataclass
class SpeculativeStat:
    img_idx: int
    wall_clock_s: float
    iters: int
    residual_history: list[list[float]]
    best_draft_indices_history: list[list[int]]


def run(num_images: int = NUM_IMAGES, force: bool = False) -> None:
    available = set(get_available_models())
    x, y, y_null = make_eval_batch(num_images)

    for pair in tqdm(SPECULATIVE_CONFIGS, desc="speculative pairs"):
        if pair["draft"] not in available or pair["base"] not in available:
            continue
        base_model = load_model(pair["base"])
        draft_model = load_model(pair["draft"])

        for num_steps in pair["num_steps"]:
            for threshold in pair["thresholds"]:
                for overlap in [True, False]:
                    eval_key = (
                        f"speculative_draft_{pair['draft']}_base_{pair['base']}"
                        f"_steps_{num_steps}_threshold_{threshold}_draft_k_{pair['spec_k']}"
                        f"_{'overlap' if overlap else 'sequential'}"
                    )
                    if not force and result_exists(SPEC_NAME, eval_key):
                        continue

                    records = []
                    for idx in tqdm(range(num_images), desc=eval_key, leave=False):
                        t0 = time.perf_counter()
                        _, stats = speculative_trajectory(
                            base_model,
                            draft_model,
                            x[idx],
                            y[idx],
                            y_null[idx],
                            num_steps,
                            pair["spec_k"],
                            CFG_SCALE,
                            threshold,
                            overlap=overlap,
                        )
                        records.append(
                            asdict(
                                SpeculativeStat(
                                    img_idx=idx,
                                    wall_clock_s=time.perf_counter() - t0,
                                    iters=stats.iters,
                                    residual_history=stats.residual_history,
                                    best_draft_indices_history=stats.best_draft_indices_history,
                                )
                            )
                        )

                    save_result(SPEC_NAME, eval_key, records)


if __name__ == "__main__":
    run()
