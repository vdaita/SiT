"""
run_flux2_experiments.py
========================
Evaluates Picard, Two-Picard, and Speculative trajectory methods
on the FLUX.2 klein model family.

Draft: black-forest-labs/FLUX.2-klein-base-4B  (4B, undistilled, Apache 2.0)
Base:  black-forest-labs/FLUX.2-klein-base-9B  (9B, undistilled)

Both models are undistilled rectified flow transformers sharing the same
VAE (AutoencoderKLFlux2) and text encoder (Qwen3). Text encoders and VAE
are loaded once from the base pipeline and reused for the draft transformer.

Requires diffusers from git main:
  pip install git+https://github.com/huggingface/diffusers.git

Results:
  outputs/results_flux2_baseline.json
  outputs/results_flux2_two_picard.json
  outputs/results_flux2_speculative.json

Images:
  outputs/images/flux2_{baseline,two_picard,spec}_<key>/img_NNN.png
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from diffusers import Flux2KleinPipeline
from diffusers.models import Flux2Transformer2DModel
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import picard_trajectory, two_picard_trajectory, speculative_trajectory

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.bfloat16

BASE_MODEL_ID   = "black-forest-labs/FLUX.2-klein-base-9B"
DRAFT_MODEL_ID  = "black-forest-labs/FLUX.2-klein-base-4B"

IMAGE_SIZE      = 512
LATENT_SIZE     = IMAGE_SIZE // 8
LATENT_CHANNELS = 16
CFG_SCALE       = 4.0
BATCH_SIZE      = 16
SEED            = 42
NUM_IMAGES      = 32

NUM_STEPS_SWEEP  = [16, 28]
THRESHOLDS       = [0.05, 0.1]
DRAFT_INIT_SWEEP = [1, 2, 4]
SPEC_K           = 4

EVAL_PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a beautiful sunset over the ocean",
    "a futuristic city skyline at night",
    "a bowl of fresh fruit on a wooden table",
    "a mountain landscape with snow-capped peaks",
    "a portrait of a person smiling",
    "a red sports car on a winding road",
    "a cozy coffee shop interior",
    "a field of sunflowers under blue sky",
    "an abstract painting with vibrant colors",
    "a dog playing in the park",
    "a plate of pasta with tomato sauce",
    "a waterfall in a tropical forest",
    "a bookshelf filled with colorful books",
    "a lighthouse on a rocky cliff",
    "a hot air balloon over a valley",
    "a snowy mountain cabin at dusk",
    "a bustling night market in Asia",
    "a serene Japanese garden in autumn",
    "a medieval castle on a hilltop",
    "a colorful coral reef underwater",
    "a steam train crossing a stone bridge",
    "a desert landscape with sand dunes",
    "a northern lights display over a frozen lake",
    "a vintage bicycle leaning against a brick wall",
    "a golden wheat field at harvest time",
    "a glowing neon sign in a rainy city",
    "a peaceful beach at sunrise",
    "a dense forest path in morning fog",
    "an old lighthouse beam cutting through night fog",
    "a hummingbird hovering over a red flower",
    "a child flying a kite in a meadow",
]

RESULTS_DIR   = "outputs"
IMAGES_DIR    = "outputs/images"
RESULTS_FILES = {
    "baseline":    "outputs/results_flux2_baseline.json",
    "two_picard":  "outputs/results_flux2_two_picard.json",
    "speculative": "outputs/results_flux2_speculative.json",
}


# ---------------------------------------------------------------------------
# FLUX.2 Model Wrapper
#
# Flux2Transformer2DModel.forward() signature:
#   (hidden_states, timestep, encoder_hidden_states,
#    img_ids, txt_ids, guidance, return_dict)
#
# hidden_states must be a packed sequence: (batch, seq_len, in_channels)
# where in_channels = LATENT_CHANNELS * patch_size^2 = 16 * 2 * 2 * 2 = 128
# (FLUX.2 packs 2x2 spatial patches AND doubles channels via a 2x channel
# grouping, yielding 128ch tokens from 16ch latents at patch_size=2).
#
# We pack on entry and unpack on exit so the rest of the codebase sees
# the normal (batch, 16, H, W) spatial format throughout.
# ---------------------------------------------------------------------------
class Flux2ModelWrapper(nn.Module):
    def __init__(self, transformer, prompt_embeds, null_prompt_embeds,
                 latent_h, latent_w):
        super().__init__()
        self.transformer = transformer
        self.latent_h = latent_h
        self.latent_w = latent_w

        self.register_buffer("prompt_embeds", prompt_embeds)
        self.register_buffer("null_prompt_embeds", null_prompt_embeds)

        # img_ids over packed spatial grid: (H//2) x (W//2) tokens
        self.register_buffer(
            "img_ids",
            self._make_img_ids(latent_h // 2, latent_w // 2)
        )

    def _make_img_ids(self, h, w):
        img_ids = torch.zeros(h, w, 4)
        img_ids[:, :, 1] = torch.arange(h).unsqueeze(1)
        img_ids[:, :, 2] = torch.arange(w).unsqueeze(0)
        # img_ids[:, :, 3] left as 0 — time/scale dimension, 0 is correct for images
        return img_ids.reshape(-1, 4)

    @staticmethod
    def _pack(x):
        """(batch, 16, H, W) -> (batch, H//2 * W//2, 128)"""
        batch, c, h, w = x.shape
        x = x.reshape(batch, c, h // 2, 2, w // 2, 2)   # split spatial into 2x2 patches
        x = x.permute(0, 2, 4, 1, 3, 5)                  # (batch, h//2, w//2, c, 2, 2)
        x = x.reshape(batch, (h // 2) * (w // 2), c * 4) # (batch, seq, 64)
        # FLUX.2 x_embedder expects 128ch: pack pairs of sequence tokens
        # by interleaving adjacent spatial positions along the sequence dim
        seq = x.shape[1]
        x = x.reshape(batch, seq // 2, c * 8)             # (batch, seq//2, 128)
        return x

    @staticmethod
    def _unpack(x, h, w):
        """(batch, seq//2, 128) -> (batch, 16, H, W)"""
        batch, seq_half, c_packed = x.shape
        c = c_packed // 8
        x = x.reshape(batch, seq_half * 2, c * 4)         # (batch, seq, 64)
        x = x.reshape(batch, h // 2, w // 2, c, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(batch, c, h, w)
        return x

    def forward(self, x, t, y):
        """
        x: (batch, 16, H, W)
        t: (batch,) timesteps in [0, 1]
        y: (batch,) prompt indices, negative = null conditioning
        Returns: velocity v, same shape as x
        """
        batch, c, h, w = x.shape

        enc_list = []
        for i in range(batch):
            idx = int(y[i].item())
            if idx < 0:
                enc_list.append(self.null_prompt_embeds[0])
            else:
                enc_list.append(self.prompt_embeds[idx % self.prompt_embeds.shape[0]])

        encoder_hidden = torch.stack(enc_list, dim=0)

        txt_ids = torch.zeros(
            encoder_hidden.shape[1], 4,
            device=x.device, dtype=x.dtype
        )
        img_ids  = self.img_ids.to(x.device, x.dtype)
        guidance = torch.full((batch,), CFG_SCALE, device=x.device, dtype=x.dtype)

        x_packed = self._pack(x)

        out_packed = self.transformer(
            hidden_states=x_packed,
            timestep=t,
            encoder_hidden_states=encoder_hidden,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]

        return self._unpack(out_packed, h, w)


# ---------------------------------------------------------------------------
# Prompt encoding
# ---------------------------------------------------------------------------
def encode_prompts(pipe, prompts):
    all_enc = []
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Encoding prompts"):
            result = pipe.encode_prompt(
                prompt=prompt,
                device=DEVICE,
                num_images_per_prompt=1,
            )
            all_enc.append(result[0])
    return torch.cat(all_enc, dim=0)


def encode_null_prompt(pipe):
    with torch.no_grad():
        result = pipe.encode_prompt(
            prompt="",
            device=DEVICE,
            num_images_per_prompt=1,
        )
    return result[0][:1]


# ---------------------------------------------------------------------------
# Decode latents
# ---------------------------------------------------------------------------
def decode(pipe, latent):
    latent_scaled = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return pipe.vae.decode(latent_scaled).sample


def save_img(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor, path, nrow=1, normalize=True, value_range=(-1, 1))


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading base pipeline: {BASE_MODEL_ID}")
    base_pipe = Flux2KleinPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    base_pipe.set_progress_bar_config(disable=True)

    # Encode all prompts once using base pipeline text encoder
    print("Encoding evaluation prompts...")
    prompts = EVAL_PROMPTS[:NUM_IMAGES]
    prompt_embeds = encode_prompts(base_pipe, prompts)
    null_embeds   = encode_null_prompt(base_pipe)

    latent_h = LATENT_SIZE
    latent_w = LATENT_SIZE

    # Wrap base transformer
    base_transformer = base_pipe.transformer
    base_model = Flux2ModelWrapper(
        base_transformer,
        prompt_embeds, null_embeds,
        latent_h, latent_w,
    ).to(DEVICE)
    base_model.eval()

    # Load draft transformer only — reuse text encoder and VAE from base pipe
    print(f"Loading draft transformer: {DRAFT_MODEL_ID}")
    draft_transformer = Flux2Transformer2DModel.from_pretrained(
        DRAFT_MODEL_ID,
        subfolder="transformer",
        torch_dtype=DTYPE,
    ).to(DEVICE)
    draft_model = Flux2ModelWrapper(
        draft_transformer,
        prompt_embeds, null_embeds,
        latent_h, latent_w,
    ).to(DEVICE)
    draft_model.eval()

    # Fixed noise and conditioning
    torch.manual_seed(SEED)
    all_x = [
        torch.randn(BATCH_SIZE, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
                    device=DEVICE, dtype=DTYPE)
        for _ in range(NUM_IMAGES)
    ]
    all_y      = [torch.tensor([i % len(prompts)] * BATCH_SIZE, device=DEVICE)
                  for i in range(NUM_IMAGES)]
    all_y_null = [torch.full((BATCH_SIZE,), -1, device=DEVICE)
                  for _ in range(NUM_IMAGES)]

    # -----------------------------------------------------------------------
    # 1. Baseline
    # -----------------------------------------------------------------------
    baseline = load_results("baseline")
    print("\n=== Baseline (Picard) ===")

    for num_steps in NUM_STEPS_SWEEP:
        for tau in THRESHOLDS:
            key = f"flux2_steps{num_steps}_tau{tau}"
            if key in baseline:
                print(f"  Skipping {key}")
                continue

            print(f"\n[baseline] {key}")
            records = []

            for idx in tqdm(range(NUM_IMAGES), desc=key, leave=False):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out, stats = picard_trajectory(
                        base_model, all_x[idx], all_y[idx], all_y_null[idx],
                        num_steps, CFG_SCALE, tau,
                    )
                wall = time.perf_counter() - t0

                img = decode(base_pipe, out)
                save_img(img, f"{IMAGES_DIR}/flux2_baseline_{key}/img_{idx:03d}.png")

                records.append({
                    "img_idx":      idx,
                    "prompt":       prompts[idx % len(prompts)],
                    "wall_clock_s": wall,
                    "iters":        stats["iters"],
                    "residual":     stats["residual"],
                })

            baseline[key] = records
            save_results("baseline", baseline)

    # -----------------------------------------------------------------------
    # 2. Two-Picard
    # -----------------------------------------------------------------------
    two_picard = load_results("two_picard")
    print("\n=== Two-Picard ===")

    for num_steps in NUM_STEPS_SWEEP:
        for tau in THRESHOLDS:
            for n_draft in DRAFT_INIT_SWEEP:
                key = f"flux2_steps{num_steps}_tau{tau}_dinit{n_draft}"
                if key in two_picard:
                    print(f"  Skipping {key}")
                    continue

                print(f"\n[two_picard] {key}")
                records = []

                for idx in tqdm(range(NUM_IMAGES), desc=key, leave=False):
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out, stats = two_picard_trajectory(
                            base_model, draft_model,
                            all_x[idx], all_y[idx], all_y_null[idx],
                            num_steps, n_draft, CFG_SCALE, tau, tau,
                        )
                    wall = time.perf_counter() - t0

                    img = decode(base_pipe, out)
                    save_img(img, f"{IMAGES_DIR}/flux2_two_picard_{key}/img_{idx:03d}.png")

                    records.append({
                        "img_idx":      idx,
                        "prompt":       prompts[idx % len(prompts)],
                        "wall_clock_s": wall,
                        "draft_iters":  stats["draft_iters"],
                        "base_iters":   stats["base_iters"],
                        "total_iters":  stats["draft_iters"] + stats["base_iters"],
                        "residual":     stats["residual"],
                    })

                two_picard[key] = records
                save_results("two_picard", two_picard)

    # -----------------------------------------------------------------------
    # 3. Speculative
    # -----------------------------------------------------------------------
    speculative = load_results("speculative")
    print("\n=== Speculative ===")

    for num_steps in NUM_STEPS_SWEEP:
        for tau in THRESHOLDS:
            for overlap in [False, True]:
                overlap_str = "overlap" if overlap else "sequential"
                key = f"flux2_steps{num_steps}_tau{tau}_K{SPEC_K}_{overlap_str}"
                if key in speculative:
                    print(f"  Skipping {key}")
                    continue

                print(f"\n[speculative] {key}")
                records = []

                for idx in tqdm(range(NUM_IMAGES), desc=key, leave=False):
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out, stats = speculative_trajectory(
                            base_model, draft_model,
                            all_x[idx], all_y[idx], all_y_null[idx],
                            num_steps, SPEC_K, CFG_SCALE, tau,
                            overlap=overlap,
                        )
                    wall = time.perf_counter() - t0

                    img = decode(base_pipe, out)
                    save_img(img, f"{IMAGES_DIR}/flux2_spec_{key}/img_{idx:03d}.png")

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
                        "prompt":             prompts[idx % len(prompts)],
                        "wall_clock_s":       wall,
                        "iters":              stats["iters"],
                        "residual":           stats["residual"],
                        "acceptance_rate":    acceptance_rate,
                        "acceptance_history": acc_hist,
                    })

                speculative[key] = records
                save_results("speculative", speculative)

    print("\nDone. Results in outputs/results_flux2_*.json")


if __name__ == "__main__":
    with torch.no_grad():
        main()