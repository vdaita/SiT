"""
run_experiments.py
==================
Edit MODEL_PAIRS below to configure each draft/base pair with its own
spec_k, thresholds, and num_steps. Baseline uses global NUM_STEPS_SWEEP
and THRESHOLDS. Two-picard uses per-pair thresholds/steps + global
DRAFT_INIT_SWEEP.

Results (separate files):
  outputs/results_baseline.json
  outputs/results_two_picard.json
  outputs/results_speculative.json

Images:
  outputs/images/{baseline,two_picard,spec}_<key>/img_NNN.png
"""

import argparse
import json
import os
import sys
import time

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import SiT_XL_2, SiT_L_2, SiT_B_2, SiT_S_2
from download import find_model
from inference import picard_trajectory, two_picard_trajectory, speculative_trajectory

# ---------------------------------------------------------------------------
# Global sweeps (used by baseline and two_picard)
# ---------------------------------------------------------------------------
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE       = 256
LATENT_SIZE      = IMAGE_SIZE // 8
NUM_CLASSES      = 1000
CFG_SCALE        = 4.0
BATCH_SIZE       = 1
SEED             = 42
NUM_IMAGES       = 32

NUM_STEPS_SWEEP  = [8, 16, 32]
THRESHOLDS       = [0.01, 0.05, 0.1]
DRAFT_INIT_SWEEP = [1, 2, 4, 8]

# ---------------------------------------------------------------------------
# Per-pair config for speculative (and optionally two_picard).
# Each entry can override:
#   spec_k       — draft steps per base forward pass (required)
#   thresholds   — overrides global THRESHOLDS for this pair
#   num_steps    — overrides global NUM_STEPS_SWEEP for this pair
# ---------------------------------------------------------------------------
MODEL_PAIRS = [
    {
        "draft": "S", "base": "B", "spec_k": 4,
        "thresholds": [0.05, 0.1],
        "num_steps":  [16, 32],
    },
    {
        "draft": "S", "base": "L", "spec_k": 4,
        "thresholds": [0.05, 0.1],
        "num_steps":  [16, 32],
    },
    {
        "draft": "B", "base": "L", "spec_k": 2,
        "thresholds": [0.05, 0.1],
        "num_steps":  [16, 32],
    }
]

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
RESULTS_DIR   = "outputs"
IMAGES_DIR    = "outputs/images"
RESULTS_FILES = {
    "baseline":    "outputs/results_baseline.json",
    "two_picard":  "outputs/results_two_picard.json",
    "speculative": "outputs/results_speculative.json",
}
MODEL_REGISTRY = {
    "S":  (SiT_S_2,  "models/S.pt"),
    "B":  (SiT_B_2,  "models/B.pt"),
    "L":  (SiT_L_2,  "models/L.pt"),
    # "XL": (SiT_XL_2, "models/XL.pt"),
}

def load_model(name):
    cls, ckpt = MODEL_REGISTRY[name]
    m = cls(input_size=LATENT_SIZE).to(DEVICE)
    m.load_state_dict(find_model(ckpt), strict=True)
    m.eval()
    return m

def load_results(kind):
    path = RESULTS_FILES[kind]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_results(kind, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILES[kind], "w") as f:
        json.dump(data, f, indent=2)

def save_img(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor, path, nrow=1, normalize=True, value_range=(-1, 1))

def decode(vae, latent):
    return vae.decode(latent / 0.18215).sample

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["S", "B", "L", "XL"])
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    skip      = not args.no_skip
    available = [m for m in args.models
                 if m in MODEL_REGISTRY and os.path.exists(MODEL_REGISTRY[m][1])]
    print(f"Available models: {available}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)

    torch.manual_seed(SEED)
    all_x      = [torch.randn(BATCH_SIZE, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
                  for _ in range(args.num_images)]
    all_y      = [torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)
                  for _ in range(args.num_images)]
    all_y_null = [torch.full((BATCH_SIZE,), NUM_CLASSES, device=DEVICE)
                  for _ in range(args.num_images)]

    pairs = [p for p in MODEL_PAIRS
             if p["draft"] in available and p["base"] in available]

    # -----------------------------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------------------------
    baseline = load_results("baseline")

    for model_name in tqdm(available, desc="Baseline"):
        for num_steps in NUM_STEPS_SWEEP:
            for tau in THRESHOLDS:
                key = f"{model_name}_steps{num_steps}_tau{tau}"
                if skip and key in baseline:
                    continue

                print(f"\n[baseline] {key}")
                model   = load_model(model_name)
                records = []

                for idx in tqdm(range(args.num_images), desc=key, leave=False):
                    t0 = time.perf_counter()
                    out, stats = picard_trajectory(
                        model, all_x[idx], all_y[idx], all_y_null[idx],
                        num_steps, CFG_SCALE, tau,
                    )
                    wall = time.perf_counter() - t0
                    save_img(decode(vae, out),
                             f"{IMAGES_DIR}/baseline_{key}/img_{idx:03d}.png")
                    records.append({
                        "img_idx":      idx,
                        "wall_clock_s": wall,
                        "iters":        stats["iters"],
                        "residual":     stats["residual"],
                    })

                baseline[key] = records
                save_results("baseline", baseline)
                del model
                torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. Two-picard  (uses per-pair thresholds/steps)
    # -----------------------------------------------------------------------
    two_picard = load_results("two_picard")

    for pair in tqdm(pairs, desc="Two-picard"):
        base_name  = pair["base"]
        draft_name = pair["draft"]
        p_steps    = pair.get("num_steps",  NUM_STEPS_SWEEP)
        p_taus     = pair.get("thresholds", THRESHOLDS)

        for num_steps in p_steps:
            for tau in p_taus:
                for n_draft in DRAFT_INIT_SWEEP:
                    key = f"{draft_name}to{base_name}_steps{num_steps}_tau{tau}_dinit{n_draft}"
                    if skip and key in two_picard:
                        continue

                    print(f"\n[two_picard] {key}")
                    base_model  = load_model(base_name)
                    draft_model = load_model(draft_name)
                    records     = []

                    for idx in tqdm(range(args.num_images), desc=key, leave=False):
                        t0 = time.perf_counter()
                        out, stats = two_picard_trajectory(
                            base_model, draft_model,
                            all_x[idx], all_y[idx], all_y_null[idx],
                            num_steps, n_draft, CFG_SCALE, tau, tau,
                        )
                        wall = time.perf_counter() - t0
                        save_img(decode(vae, out),
                                 f"{IMAGES_DIR}/two_picard_{key}/img_{idx:03d}.png")
                        records.append({
                            "img_idx":      idx,
                            "wall_clock_s": wall,
                            "draft_iters":  stats["draft_iters"],
                            "base_iters":   stats["base_iters"],
                            "total_iters":  stats["draft_iters"] + stats["base_iters"],
                            "residual":     stats["residual"],
                        })

                    two_picard[key] = records
                    save_results("two_picard", two_picard)
                    del base_model, draft_model
                    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 3. Speculative  (uses per-pair spec_k, thresholds, num_steps)
    # -----------------------------------------------------------------------
    speculative = load_results("speculative")

    for pair in tqdm(pairs, desc="Speculative"):
        base_name  = pair["base"]
        draft_name = pair["draft"]
        K          = pair["spec_k"]
        p_steps    = pair.get("num_steps",  NUM_STEPS_SWEEP)
        p_taus     = pair.get("thresholds", THRESHOLDS)

        for num_steps in p_steps:
            for tau in p_taus:
                for overlap in [True, False]:
                    overlap_str = "overlap" if overlap else "sequential"
                    key = (f"{draft_name}to{base_name}_steps{num_steps}"
                           f"_tau{tau}_K{K}_{overlap_str}")
                    if skip and key in speculative:
                        print("Key ", key, " in speculative")
                        continue

                    print(f"\n[speculative] {key}")
                    base_model  = load_model(base_name)
                    draft_model = load_model(draft_name)
                    records     = []

                    for idx in tqdm(range(args.num_images), desc=key, leave=False):
                        t0 = time.perf_counter()
                        out, stats = speculative_trajectory(
                            base_model, draft_model,
                            all_x[idx], all_y[idx], all_y_null[idx],
                            num_steps, K, CFG_SCALE, tau,
                            overlap=overlap,
                        )
                        wall = time.perf_counter() - t0
                        save_img(decode(vae, out),
                                 f"{IMAGES_DIR}/spec_{key}/img_{idx:03d}.png")

                        acc_hist = (
                            [int(v) for v in
                             torch.cat(stats["best_draft_indices_history"]).tolist()]
                            if stats["best_draft_indices_history"] else []
                        )
                        acceptance_rate = (
                            sum(1 for a in acc_hist if a > 0) / len(acc_hist)
                            if acc_hist else 0.0
                        )
                        records.append({
                            "img_idx":            idx,
                            "wall_clock_s":       wall,
                            "iters":              stats["iters"],
                            "residual":           stats["residual"],
                            "acceptance_rate":    acceptance_rate,
                            "acceptance_history": acc_hist,
                            "draft_residual_grid": [
                                r.tolist() for r in stats["draft_residual_grid_history"]
                            ],
                        })

                    speculative[key] = records
                    save_results("speculative", speculative)
                    del base_model, draft_model
                    torch.cuda.empty_cache()

    print("\nDone. Results in outputs/results_*.json")


if __name__ == "__main__":
    with torch.no_grad():
        main()