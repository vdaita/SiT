from typing import TypedDict, Dict, Tuple
from inference import compute_threshold_schedule
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

from models import SiT_L_2, SiT_B_2, SiT_S_2
from inference import picard_trajectory, two_picard_trajectory, speculative_trajectory, two_picard_trajectory_grid
from torchmetrics.image.kid import KernelInceptionDistance

from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = 256
latent_size = image_size // 8
num_classes = 1000
cfg_scale = 4.0
batch_size = 1
seed = 42
num_images = 32

num_steps_sweep = [8, 16, 32]
thresholds = [0.01, 0.05, 0.1]
draft_init_sweep = [1, 2, 4, 8]
baseline_models = ["S", "B"]

class ThresholdPair(TypedDict):
    draft: float
    base: float

class SpeculativeConfig(TypedDict):
    draft: str
    base: str
    thresholds: List[ThresholdPair]

two_picard_speculative_configs: List[SpeculativeConfig] = [
    {
        "draft": "S",
        "base": "B",
        "thresholds": [
            {"draft": 0.05, "base": 0.05},
            {"draft": 0.05, "base": 0.1},
            {"draft": 0.1, "base": 0.1},
            {"draft": 0.1, "base": 0.2}
        ],
    },
    {
        "draft": "S",
        "base": "L",
        "thresholds": [
            {"draft": 0.05, "base": 0.05},
            {"draft": 0.05, "base": 0.1},
            {"draft": 0.1, "base": 0.1},
            {"draft": 0.1, "base": 0.2}
        ]
    }
]

speculative_configs = [
    {
        "draft": "S", "base": "B", "spec_k": 4,
        "num_steps": [16, 32]
    },
    {
        "draft": "S", "base": "L", "spec_k": 4,
        "num_steps": [16, 32]
    },
    {
        "draft": "B", "base": "L", "spec_k": 2,
        "num_steps": [16, 32]
    }
]

@dataclass
class BaselineStat:
    img_idx: int
    wall_clock_s: float
    iters: int
    residual: List[float]
    threshold: float

@dataclass
class TwoPicardGridStat:
    draft_iters: int
    base_iters: int
    kid: int

@dataclass
class TwoPicardTimingStat:
    img_idx: int
    wall_clock_s: float
    draft_iters: int
    base_iters: int

@dataclass
class SpeculativeStat:
    img_idx: int
    wall_clock_s: float
    iters: int
    acceptance_rate: List[float]
    draft_residual_grid: List[List[float]]


if __name__ == "__main__":
    with torch.no_grad():
        # load in the images for evaluation
        x = [torch.randn(batch_size, 4, latent_size, latent_size, device=device) for _ in range(num_images)]
        y = [torch.randint(0, num_classes, (batch_size,), device=device)]
        y_null = [torch.full((batch_size,), num_classes, device=device)]
        
        # baseline run
        for model_id in baseline_models:
            model = models[model_id]
            for num_steps in num_steps_sweep:
                for idx in tqdm(range(num_images), desc=f"baseline num_steps={num_steps}", leave=True):
                    eval_key = f"baseline_model_{model_id}_num_steps_{num_steps}_image_{idx}"
                    if record_exists(eval_key):
                        continue

                    out, stats = picard_trajectory(
                        model, x[idx], y[idx], y_null[idx],
                        num_steps, cfg_scale, thresholds
                    )
                    for threshold_idx, threshold in enumerate(thresholds):
                        record = BaselineStat(
                            img_idx=idx,
                            wall_clock_s=stats.durations[threshold_idx],
                            iters=stats.iters[threshold_idx],
                            residual=stats.residual_history[threshold_idx - 1],
                            threshold=threshold
                        )
                        save_record(record)
            
        # two picard grid predictions
        

            # figure out the KID of these images

        # two-picard standard evaluations
        two_picard = ...
        for pair in tqdm(model_pairs, desc="two picard"):
            for num_steps in pair["num_steps"]:
                for base_threshold in pair["base_thresholds"]:
                    for draft_threshold in pair["draft_thresholds"]:
                        key = f""
                        records = []
                        for idx in tqdm(range(num_images), desc=key):
                            t0 = time.perf_counter()
                            out, stats = two_picard_trajectory(
                                base_model, draft_model,
                                x[idx], y[idx], y_null[idx],
                                num_steps, n_draft, cfg_scale, base_threshold, draft_threshold
                            )
                            wall = time.perf_counter() - t0
                            records.append({
                                "img_idx": idx,
                                "wall_clock_s": wall,
                                "draft_iters": stats["draft_iters"],
                                "base_iters": stats["base_iters"],
                                "residual": stats["residual"]
                            })
                        two_picard[key] = records
                        torch.cuda.empty_cache()
        
        # speculative trajectory
        speculative = ...
        for pair in tqdm(model_pairs, desc="speculative"):
            for num_steps in pair["num_steps"]:
                for thresholds in pair["thresholds"]:
                    for overlap in [True, False]:
                        key = f"{draft_name}_to_{base_name}_steps{num_steps}_threshold_{threshold}_draft_k_{draft_k}" + ("_overlap" if overlap else "")
                        base_model = load_model(base_name)
                        draft_model = load_model(draft_name)
                        records = []
                        
                        for idx in tqdm(range(num_images), desc=key):
                            t0 = time.perf_counter()
                            out, stats = speculative_trajectory(
                                base_model, draft_model,
                                x[idx], y[idx], y_null[idx],
                                num_steps, K, cfg_scale, threshold,
                                overlap=overlap
                            )
                            wall = time.perf_counter() - t0
                            save_image()
                            records.append()