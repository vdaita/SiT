"""
Simplified Flux2 Klein inference with three self-contained functions,
plus Picard trajectory solvers (single-model and two-stage draft+base).

Functions:
  1. encode_text:   text -> embedding
  2. denoise_step:  embedding + latent + (sigma, sigma_next) -> updated latent
  3. decode_latent: final latent -> PIL image

  + picard_trajectory:       single-model Picard fixed-point iteration
  + two_picard_trajectory:   two-stage draft+base Picard iteration

NOTE: The 9B base model and 4B draft model use DIFFERENT text encoders.
      Text must be encoded separately for each model.

The __main__ block runs comprehensive benchmarks across prompts, thresholds,
and draft configurations, saving results and generating comparison plots.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Plotting will be skipped.")


# ──────────────────────────────────────────────────────────────────────
# Benchmark prompts
# ──────────────────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = [
    "A cat holding a sign that says hello world",
    "A photorealistic portrait of an astronaut riding a horse on Mars",
    "An oil painting of a sunflower field at sunset in the style of Van Gogh",
    "A futuristic cyberpunk cityscape with neon lights reflecting in rain puddles",
    "A cute robot reading a book in a cozy library",
    "A macro photograph of a dewdrop on a spider web at dawn",
    "A watercolor painting of a Japanese garden with cherry blossoms",
    "An ancient dragon sleeping on a pile of gold coins in a cave",
    "A minimalist flat design illustration of a mountain landscape",
    "A steampunk airship flying over Victorian London at twilight",
    "A bowl of ramen with steam rising, photographed from above",
    "A double exposure photograph of a wolf and a forest",
    "An isometric pixel art scene of a tiny medieval village",
    "A Renaissance-style painting of a programmer debugging code",
    "A bioluminescent deep-sea creature in the Mariana Trench",
]


# ──────────────────────────────────────────────────────────────────────
# Helpers (geometry / packing)
# ──────────────────────────────────────────────────────────────────────


def _prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    B, L, _ = x.shape
    out = []
    for _ in range(B):
        coords = torch.cartesian_prod(
            torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(L)
        )
        out.append(coords)
    return torch.stack(out)


def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    B, _, H, W = latents.shape
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(H), torch.arange(W), torch.arange(1)
    )
    return ids.unsqueeze(0).expand(B, -1, -1)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    B, C4, H2, W2 = latents.shape
    C = C4 // 4
    return (
        latents.reshape(B, C, 2, 2, H2, W2)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(B, C, H2 * 2, W2 * 2)
    )


def _pack(latents: torch.Tensor) -> torch.Tensor:
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


def _unpack_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
    out_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = h_ids.max() + 1
        w = w_ids.max() + 1
        flat = h_ids * w + w_ids
        buf = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        buf.scatter_(0, flat.unsqueeze(1).expand(-1, ch), data)
        out_list.append(buf.view(h, w, ch).permute(2, 0, 1))
    return torch.stack(out_list, dim=0)


# ──────────────────────────────────────────────────────────────────────
# Scheduling helpers
# ──────────────────────────────────────────────────────────────────────


def compute_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def build_sigma_schedule(
    num_steps: int,
    mu: float,
    num_train_timesteps: int = 1000,
) -> np.ndarray:
    raw_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps).astype(np.float64)
    exp_mu = math.exp(mu)
    shifted = exp_mu / (exp_mu + (1.0 / raw_sigmas - 1.0))
    return np.append(shifted, 0.0).astype(np.float32)


def euler_step(
    latents: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    dt = sigma_next - sigma
    prev = latents.float() + dt * model_output.float()
    return prev.to(model_output.dtype)


# ──────────────────────────────────────────────────────────────────────
# Function 1: Text -> Embedding
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def encode_text(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompt: str,
    device: torch.device,
    max_sequence_length: int = 512,
    hidden_layers: tuple[int, ...] = (9, 18, 27),
) -> tuple[torch.Tensor, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack(
        [output.hidden_states[k] for k in hidden_layers], dim=1
    )
    stacked = stacked.to(dtype=text_encoder.dtype, device=device)
    B, nl, seq_len, hd = stacked.shape
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(B, seq_len, nl * hd)
    text_ids = _prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


def encode_all_prompts(
    model_id: str,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    label: str = "",
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Load the text encoder for a given model, encode all prompts + the empty
    negative prompt, then free the encoder.

    The 9B and 4B models have different text encoders (different architectures,
    hidden layers, tokenizers), so this must be called once per model.

    Returns:
        (all_prompt_embeds, all_text_ids, neg_prompt_embeds, neg_text_ids)
    """
    tag = f" ({label})" if label else ""
    print(f"\nLoading tokenizer{tag} from {model_id}...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_id, subfolder="tokenizer"
    )
    print(f"Loading text encoder{tag} from {model_id}...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)

    all_pe, all_ti = [], []
    print(f"Encoding {len(prompts)} prompts{tag}...")
    for i, prompt in enumerate(prompts):
        pe, ti = encode_text(text_encoder, tokenizer, prompt, device)
        all_pe.append(pe)
        all_ti.append(ti)
        if (i + 1) % 5 == 0 or i == 0 or i == len(prompts) - 1:
            print(
                f"  [{i+1}/{len(prompts)}] \"{prompt[:55]}\" -> {pe.shape}"
            )

    print(f"Encoding negative (empty) prompt{tag}...")
    neg_pe, neg_ti = encode_text(text_encoder, tokenizer, "", device)

    del text_encoder, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Text encoder{tag} freed.")
    return all_pe, all_ti, neg_pe, neg_ti


# ──────────────────────────────────────────────────────────────────────
# Function 2: Single Euler denoise step
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def denoise_step(
    transformer: Flux2Transformer2DModel,
    latents: torch.Tensor,
    latent_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    sigma: float,
    sigma_next: float,
    num_train_timesteps: int = 1000,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_text_ids: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    t_val = sigma * num_train_timesteps
    t_expanded = torch.full(
        (latents.shape[0],), t_val, device=latents.device, dtype=latents.dtype
    )
    latent_input = latents.to(transformer.dtype)

    with transformer.cache_context("cond"):
        noise_pred = transformer(
            hidden_states=latent_input,
            timestep=t_expanded / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            return_dict=False,
        )[0]
    noise_pred = noise_pred[:, : latents.size(1)]

    if do_cfg:
        with transformer.cache_context("uncond"):
            neg_pred = transformer(
                hidden_states=latent_input,
                timestep=t_expanded / 1000,
                guidance=None,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
        neg_pred = neg_pred[:, : latents.size(1)]
        noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

    return euler_step(latents, noise_pred, sigma, sigma_next)


# ──────────────────────────────────────────────────────────────────────
# Function 3: Final latent -> PIL Image
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def decode_latent(
    vae: AutoencoderKLFlux2,
    latents: torch.Tensor,
    latent_ids: torch.Tensor,
) -> PIL.Image.Image:
    spatial = _unpack_with_ids(latents, latent_ids)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        spatial.device, spatial.dtype
    )
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(spatial.device, spatial.dtype)
    spatial = spatial * bn_std + bn_mean
    spatial = _unpatchify(spatial)
    image_tensor = vae.decode(spatial, return_dict=False)[0]
    image_tensor = image_tensor.clamp(-1.0, 1.0)
    image_tensor = (image_tensor + 1.0) / 2.0
    image_np = image_tensor[0].float().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    return PIL.Image.fromarray(image_np)


# ──────────────────────────────────────────────────────────────────────
# Shared Picard helpers
# ──────────────────────────────────────────────────────────────────────


def _model_call_cfg(
    transformer: Flux2Transformer2DModel,
    x_traj: torch.Tensor,
    sigmas_for_steps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_ids: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_text_ids: torch.Tensor | None,
    guidance_scale: float,
    num_train_timesteps: int,
) -> torch.Tensor:
    """
    Evaluate the model at all trajectory points simultaneously.
    x_traj: (T, B, seq_len, C)
    Returns: (T, B, seq_len, C) velocity predictions.
    """
    T, B, seq_len, C = x_traj.shape
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    x_flat = x_traj.reshape(T * B, seq_len, C).to(transformer.dtype)
    t_expanded = (
        (sigmas_for_steps[:, None].expand(T, B) * num_train_timesteps)
        .reshape(T * B)
        .to(x_flat.dtype)
    )

    pe = (
        prompt_embeds.unsqueeze(0)
        .expand(T, -1, -1, -1)
        .reshape(T * B, *prompt_embeds.shape[1:])
    )
    ti = (
        text_ids.unsqueeze(0)
        .expand(T, -1, -1, -1)
        .reshape(T * B, *text_ids.shape[1:])
    )
    li = (
        latent_ids.unsqueeze(0)
        .expand(T, -1, -1, -1)
        .reshape(T * B, *latent_ids.shape[1:])
    )

    with transformer.cache_context("cond"):
        v_cond = transformer(
            hidden_states=x_flat,
            timestep=t_expanded / 1000,
            guidance=None,
            encoder_hidden_states=pe,
            txt_ids=ti,
            img_ids=li,
            return_dict=False,
        )[0]
    v_cond = v_cond[:, :seq_len]

    if do_cfg:
        npe = (
            negative_prompt_embeds.unsqueeze(0)
            .expand(T, -1, -1, -1)
            .reshape(T * B, *negative_prompt_embeds.shape[1:])
        )
        nti = (
            negative_text_ids.unsqueeze(0)
            .expand(T, -1, -1, -1)
            .reshape(T * B, *negative_text_ids.shape[1:])
        )
        with transformer.cache_context("uncond"):
            v_uncond = transformer(
                hidden_states=x_flat,
                timestep=t_expanded / 1000,
                guidance=None,
                encoder_hidden_states=npe,
                txt_ids=nti,
                img_ids=li,
                return_dict=False,
            )[0]
        v_uncond = v_uncond[:, :seq_len]
        v_cond = v_uncond + guidance_scale * (v_cond - v_uncond)

    return v_cond.reshape(T, B, seq_len, C)


def compute_threshold_schedule(
    t_traj: torch.Tensor,
    threshold: float,
    num_elements: int,
    num_steps: int,
    batch_size: int,
) -> torch.Tensor:
    schedule = threshold * (0.5 + 0.5 * (1.0 - t_traj))
    return schedule


def calculate_residuals(
    x_old: torch.Tensor, x_new: torch.Tensor
) -> torch.Tensor:
    diff = (x_old.float() - x_new.float()).abs()
    return diff.reshape(diff.shape[0], -1).mean(dim=1)


def has_converged(
    residuals: torch.Tensor, thresholds: torch.Tensor
) -> bool:
    return bool((residuals <= thresholds.to(residuals.device)).all().item())


def _extract_final_from_trajectory(
    x_traj: torch.Tensor,
    v_all: torch.Tensor,
    dt_per_step: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    final_v = v_all[-1]
    final_dt = dt_per_step[-1]
    final_latents = x_traj[-1].float() + (final_dt * final_v).float()
    return final_latents.to(dtype)


def _picard_update(
    x_traj_0: torch.Tensor,
    v_all: torch.Tensor,
    dt_per_step: torch.Tensor,
) -> torch.Tensor:
    """Compute new trajectory from initial point + cumulative velocity integral."""
    v_scaled = v_all * dt_per_step[:, None, None, None]
    cumulative_v = torch.cumsum(v_scaled[:-1], dim=0)
    x_traj_new = x_traj_0.clone()
    x_traj_new[1:] = x_traj_0[1:] + cumulative_v
    return x_traj_new


# ──────────────────────────────────────────────────────────────────────
# Single-model Picard trajectory solver
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def picard_trajectory(
    transformer: Flux2Transformer2DModel,
    x_init: torch.Tensor,
    latent_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    sigmas: np.ndarray,
    num_train_timesteps: int = 1000,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_text_ids: torch.Tensor | None = None,
    guidance_scale: float = 4.0,
    threshold: float = 0.05,
    max_picard_iters: int | None = None,
    show_progress: bool = True,
    save_intermediates: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Picard fixed-point iteration for parallel-in-time denoising (single model).

    Returns:
        (final_latents, info_dict)
    """
    num_steps = len(sigmas) - 1
    B, seq_len, C = x_init.shape
    device = x_init.device
    dtype = x_init.dtype

    if max_picard_iters is None:
        max_picard_iters = num_steps

    sigmas_t = torch.tensor(sigmas[:num_steps], device=device, dtype=dtype)
    dt_per_step = torch.tensor(
        [sigmas[i + 1] - sigmas[i] for i in range(num_steps)],
        device=device,
        dtype=dtype,
    )

    threshold_schedule = compute_threshold_schedule(
        sigmas_t, threshold, B * seq_len * C, num_steps, B
    )

    x_traj_0 = x_init.unsqueeze(0).expand(num_steps, B, seq_len, C)
    x_traj = x_traj_0.clone()

    step_residuals = torch.tensor(float("inf"), device=device)
    residual_history = []
    per_step_residuals_history = []
    intermediate_latents = []
    v_all = None

    iters = 0

    for iteration in range(max_picard_iters):
        iters = iteration + 1

        v_all = _model_call_cfg(
            transformer=transformer,
            x_traj=x_traj,
            sigmas_for_steps=sigmas_t,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            latent_ids=latent_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=guidance_scale,
            num_train_timesteps=num_train_timesteps,
        )

        x_traj_new = _picard_update(x_traj_0, v_all, dt_per_step)

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        residual_history.append(float(step_residuals.max().item()))
        per_step_residuals_history.append(
            step_residuals.float().cpu().numpy().tolist()
        )

        if save_intermediates:
            iter_final = _extract_final_from_trajectory(
                x_traj_new, v_all, dt_per_step, dtype
            )
            intermediate_latents.append(iter_final.cpu())

        max_residual = float(step_residuals.max().item())
        mean_residual = float(step_residuals.mean().item())
        converged = has_converged(step_residuals, threshold_schedule)

        if show_progress:
            print(
                f"  picard iter {iters:3d}  |  "
                f"residual max={max_residual:.6f} mean={mean_residual:.6f}  |  "
                f"converged: {converged}"
            )

        x_traj = x_traj_new

        if converged:
            if show_progress:
                print(f"  Converged after {iters} iterations!")
            break

    final_latents = _extract_final_from_trajectory(
        x_traj, v_all, dt_per_step, dtype
    )

    info = {
        "iters": iters,
        "residual": float(step_residuals.max().item()),
        "residual_history": residual_history,
        "per_step_residuals": per_step_residuals_history,
        "thresholds": threshold_schedule.float().cpu().numpy().tolist(),
        "sigmas": sigmas.tolist(),
    }
    if save_intermediates:
        info["intermediate_latents"] = intermediate_latents
    return final_latents, info


# ──────────────────────────────────────────────────────────────────────
# Two-stage draft+base Picard trajectory solver
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def two_picard_trajectory(
    base_transformer: Flux2Transformer2DModel,
    draft_transformer: Flux2Transformer2DModel,
    x_init: torch.Tensor,
    latent_ids: torch.Tensor,
    # --- Base model embeddings ---
    base_prompt_embeds: torch.Tensor,
    base_text_ids: torch.Tensor,
    base_negative_prompt_embeds: torch.Tensor | None = None,
    base_negative_text_ids: torch.Tensor | None = None,
    # --- Draft model embeddings ---
    draft_prompt_embeds: torch.Tensor | None = None,
    draft_text_ids: torch.Tensor | None = None,
    draft_negative_prompt_embeds: torch.Tensor | None = None,
    draft_negative_text_ids: torch.Tensor | None = None,
    # --- Schedule / config ---
    sigmas: np.ndarray | None = None,
    num_train_timesteps: int = 1000,
    guidance_scale: float = 4.0,
    threshold: float = 0.05,
    draft_threshold: float = 0.1,
    num_draft_iters: int = 4,
    max_base_iters: int | None = None,
    show_progress: bool = True,
    save_intermediates: bool = False,
) -> tuple[torch.Tensor, dict]:
    """
    Two-stage Picard: run draft model (4B) first to warm-start the trajectory,
    then refine with the base model (9B).

    IMPORTANT: The two models use different text encoders, so separate
    prompt_embeds / text_ids must be provided for each.

    Returns:
        (final_latents, info_dict)
    """
    num_steps = len(sigmas) - 1
    B, seq_len, C = x_init.shape
    device = x_init.device
    dtype = x_init.dtype

    if max_base_iters is None:
        max_base_iters = num_steps

    sigmas_t = torch.tensor(sigmas[:num_steps], device=device, dtype=dtype)
    dt_per_step = torch.tensor(
        [sigmas[i + 1] - sigmas[i] for i in range(num_steps)],
        device=device,
        dtype=dtype,
    )

    threshold_schedule = compute_threshold_schedule(
        sigmas_t, threshold, B * seq_len * C, num_steps, B
    )
    draft_threshold_schedule = compute_threshold_schedule(
        sigmas_t, draft_threshold, B * seq_len * C, num_steps, B
    )

    x_traj_0 = x_init.unsqueeze(0).expand(num_steps, B, seq_len, C)
    x_traj = x_traj_0.clone()

    step_residuals = torch.tensor(float("inf"), device=device)
    draft_residual_history = []
    base_residual_history = []
    draft_per_step_history = []
    base_per_step_history = []
    intermediate_latents = []
    intermediate_labels = []

    draft_iters = 0
    base_iters = 0
    v_all = None

    # ── Phase 1: Draft model warm-up ─────────────────────────────────
    if show_progress:
        print(f"  --- Draft phase (max {num_draft_iters} iters, 4B model) ---")

    for iteration in range(num_draft_iters):
        draft_iters = iteration + 1

        v_all = _model_call_cfg(
            transformer=draft_transformer,
            x_traj=x_traj,
            sigmas_for_steps=sigmas_t,
            prompt_embeds=draft_prompt_embeds,
            text_ids=draft_text_ids,
            latent_ids=latent_ids,
            negative_prompt_embeds=draft_negative_prompt_embeds,
            negative_text_ids=draft_negative_text_ids,
            guidance_scale=guidance_scale,
            num_train_timesteps=num_train_timesteps,
        )

        x_traj_new = _picard_update(x_traj_0, v_all, dt_per_step)

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        draft_residual_history.append(float(step_residuals.max().item()))
        draft_per_step_history.append(
            step_residuals.float().cpu().numpy().tolist()
        )

        if save_intermediates:
            iter_final = _extract_final_from_trajectory(
                x_traj_new, v_all, dt_per_step, dtype
            )
            intermediate_latents.append(iter_final.cpu())
            intermediate_labels.append("draft")

        max_residual = float(step_residuals.max().item())
        converged = has_converged(step_residuals, draft_threshold_schedule)

        if show_progress:
            print(
                f"  draft iter {draft_iters:3d}  |  "
                f"residual max={max_residual:.6f}  |  "
                f"converged: {converged}"
            )

        x_traj = x_traj_new

        if converged:
            if show_progress:
                print(f"  Draft converged after {draft_iters} iterations!")
            break

    # ── Phase 2: Base model refinement ───────────────────────────────
    if show_progress:
        print(f"  --- Base phase (max {max_base_iters} iters, 9B model) ---")

    for iteration in range(max_base_iters):
        base_iters = iteration + 1

        v_all = _model_call_cfg(
            transformer=base_transformer,
            x_traj=x_traj,
            sigmas_for_steps=sigmas_t,
            prompt_embeds=base_prompt_embeds,
            text_ids=base_text_ids,
            latent_ids=latent_ids,
            negative_prompt_embeds=base_negative_prompt_embeds,
            negative_text_ids=base_negative_text_ids,
            guidance_scale=guidance_scale,
            num_train_timesteps=num_train_timesteps,
        )

        x_traj_new = _picard_update(x_traj_0, v_all, dt_per_step)

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        base_residual_history.append(float(step_residuals.max().item()))
        base_per_step_history.append(
            step_residuals.float().cpu().numpy().tolist()
        )

        if save_intermediates:
            iter_final = _extract_final_from_trajectory(
                x_traj_new, v_all, dt_per_step, dtype
            )
            intermediate_latents.append(iter_final.cpu())
            intermediate_labels.append("base")

        max_residual = float(step_residuals.max().item())
        converged = has_converged(step_residuals, threshold_schedule)

        if show_progress:
            print(
                f"  base  iter {base_iters:3d}  |  "
                f"residual max={max_residual:.6f}  |  "
                f"converged: {converged}"
            )

        x_traj = x_traj_new

        if converged:
            if show_progress:
                print(f"  Base converged after {base_iters} iterations!")
            break

    final_latents = _extract_final_from_trajectory(
        x_traj, v_all, dt_per_step, dtype
    )

    info = {
        "draft_iters": draft_iters,
        "base_iters": base_iters,
        "total_iters": draft_iters + base_iters,
        "residual": float(step_residuals.max().item()),
        "draft_residual_history": draft_residual_history,
        "base_residual_history": base_residual_history,
        "draft_per_step_residuals": draft_per_step_history,
        "base_per_step_residuals": base_per_step_history,
        "thresholds": threshold_schedule.float().cpu().numpy().tolist(),
        "draft_thresholds": draft_threshold_schedule.float()
        .cpu()
        .numpy()
        .tolist(),
        "sigmas": sigmas.tolist(),
    }
    if save_intermediates:
        info["intermediate_latents"] = intermediate_latents
        info["intermediate_labels"] = intermediate_labels
    return final_latents, info


# ──────────────────────────────────────────────────────────────────────
# Cost model: "equivalent sequential evaluations"
# ──────────────────────────────────────────────────────────────────────


def compute_equiv_sequential_evals(
    method: str,
    num_steps: int,
    guidance_scale: float,
    iters: int = 0,
    draft_iters: int = 0,
    base_iters: int = 0,
    draft_cost_ratio: float = 0.5,
) -> float:
    """
    Compute the number of equivalent sequential evaluations.

    Each Picard iteration evaluates all steps in parallel, so the
    sequential depth per iteration is 1 model call (or 2 with CFG).
    The draft model costs draft_cost_ratio of a base model call.

    - Euler:      num_steps * cfg_mult  (fully sequential)
    - Picard:     iters * cfg_mult      (each iter is 1 parallel batch)
    - Two-Picard: draft_iters * cfg_mult * draft_cost_ratio
                  + base_iters * cfg_mult * 1.0
    """
    cfg_mult = 2 if guidance_scale > 1.0 else 1

    if method == "euler":
        return num_steps * cfg_mult
    elif method == "picard":
        return iters * cfg_mult
    elif method == "two_picard":
        return (
            draft_iters * cfg_mult * draft_cost_ratio
            + base_iters * cfg_mult * 1.0
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────


def plot_results_from_json(results_path: str | Path, output_dir: str | Path):
    """Load results.json and generate all comparison plots."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots.")
        return

    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    experiments = results["experiments"]
    config = results["config"]

    _plot_cost_vs_draft_iters(experiments, config, output_dir)
    _plot_convergence_curves(experiments, config, output_dir)
    _plot_summary_heatmap(experiments, config, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


def _plot_cost_vs_draft_iters(experiments, config, output_dir):
    """
    THE key plot: one chart per threshold.
    X axis = number of draft iterations (0 = pure base Picard, 2, 4, 6).
    Y axis = mean equivalent sequential evaluations across prompts.
    Stacked bars: draft cost (yellow/gold) + base cost (blue).
    Horizontal line for Euler baseline.
    """
    thresholds = config["thresholds"]
    draft_steps_list = config["draft_steps_list"]
    guidance_scale = config["guidance_scale"]
    draft_cost_ratio = config["draft_cost_ratio"]
    cfg_mult = 2 if guidance_scale > 1.0 else 1

    for thresh in thresholds:
        # --- Euler baseline (threshold-independent) ---
        euler_exps = [
            e
            for e in experiments
            if e["method"] == "euler"
            and abs(e["threshold"] - thresh) < 1e-6
        ]
        euler_mean = (
            np.mean([e["equiv_sequential_evals"] for e in euler_exps])
            if euler_exps
            else None
        )

        # --- Picard (0 draft iters) ---
        picard_exps = [
            e
            for e in experiments
            if e["method"] == "picard"
            and abs(e["threshold"] - thresh) < 1e-6
        ]
        picard_base_costs = [
            e["iters"] * cfg_mult * 1.0 for e in picard_exps
        ]

        # --- Two-Picard for each draft step count ---
        draft_costs_by_ds = {}  # ds -> list of draft costs
        base_costs_by_ds = {}  # ds -> list of base costs
        for ds in draft_steps_list:
            tp_exps = [
                e
                for e in experiments
                if e["method"] == "two_picard"
                and e["num_draft_iters"] == ds
                and abs(e["threshold"] - thresh) < 1e-6
            ]
            draft_costs_by_ds[ds] = [
                e["draft_iters_actual"] * cfg_mult * draft_cost_ratio
                for e in tp_exps
            ]
            base_costs_by_ds[ds] = [
                e["base_iters_actual"] * cfg_mult * 1.0 for e in tp_exps
            ]

        # --- Build x-axis: 0, 2, 4, 6 ---
        x_labels = ["0\n(base only)"] + [str(ds) for ds in draft_steps_list]
        x_pos = np.arange(len(x_labels))

        mean_draft = [0.0]  # picard has 0 draft cost
        mean_base = [
            np.mean(picard_base_costs) if picard_base_costs else 0.0
        ]

        for ds in draft_steps_list:
            md = (
                np.mean(draft_costs_by_ds[ds])
                if draft_costs_by_ds[ds]
                else 0.0
            )
            mb = (
                np.mean(base_costs_by_ds[ds])
                if base_costs_by_ds[ds]
                else 0.0
            )
            mean_draft.append(md)
            mean_base.append(mb)

        mean_draft = np.array(mean_draft)
        mean_base = np.array(mean_base)
        mean_total = mean_draft + mean_base

        # --- Also compute per-prompt scatter ---
        all_draft_per_prompt = [[0.0] * len(picard_exps)]
        all_base_per_prompt = [picard_base_costs]
        for ds in draft_steps_list:
            all_draft_per_prompt.append(draft_costs_by_ds[ds])
            all_base_per_prompt.append(base_costs_by_ds[ds])

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))

        bar_w = 0.55
        bars_base = ax.bar(
            x_pos,
            mean_base,
            bar_w,
            label="Base model cost (9B)",
            color="#42A5F5",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        bars_draft = ax.bar(
            x_pos,
            mean_draft,
            bar_w,
            bottom=mean_base,
            label=f"Draft model cost (4B, ×{draft_cost_ratio})",
            color="#FFD54F",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        # Per-prompt scatter
        for xi in range(len(x_labels)):
            d_list = all_draft_per_prompt[xi]
            b_list = all_base_per_prompt[xi]
            if d_list and b_list:
                totals = [d + b for d, b in zip(d_list, b_list)]
                jitter = np.random.default_rng(42).uniform(
                    -0.12, 0.12, size=len(totals)
                )
                ax.scatter(
                    xi + jitter,
                    totals,
                    color="#333333",
                    alpha=0.35,
                    s=18,
                    zorder=4,
                    edgecolors="none",
                )

        # Annotate totals
        for xi in range(len(x_labels)):
            ax.text(
                xi,
                mean_total[xi] + 0.4,
                f"{mean_total[xi]:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                zorder=5,
            )

        # Euler baseline
        if euler_mean is not None:
            ax.axhline(
                euler_mean,
                color="#E53935",
                linestyle="--",
                linewidth=2,
                label=f"Euler baseline ({euler_mean:.0f})",
                zorder=2,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlabel("Number of Draft Iterations (4B model)", fontsize=13)
        ax.set_ylabel("Mean Equiv. Sequential Evaluations", fontsize=13)
        ax.set_title(
            f"Picard Cost vs. Draft Warm-up Iterations\n"
            f"(threshold = {thresh}, {config['num_prompts']} prompts, "
            f"{config['num_steps']} steps, CFG={guidance_scale})",
            fontsize=13,
        )
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        fig.savefig(
            output_dir / f"cost_vs_draft_iters_thresh_{thresh:.2f}.png",
            dpi=150,
        )
        plt.close(fig)
        print(f"  Saved: cost_vs_draft_iters_thresh_{thresh:.2f}.png")


def _plot_convergence_curves(experiments, config, output_dir):
    """
    For each threshold: one figure with subplots for each method
    (Picard-only, Two-Picard with d=2,4,6), showing residual convergence.
    Draft iterations colored differently from base iterations.
    """
    thresholds = config["thresholds"]
    draft_steps_list = config["draft_steps_list"]

    n_cols = 1 + len(draft_steps_list)

    for thresh in thresholds:
        fig, axes = plt.subplots(
            1, n_cols, figsize=(5 * n_cols, 5), sharey=True
        )
        if n_cols == 1:
            axes = [axes]

        # --- Picard-only ---
        ax = axes[0]
        picard_exps = [
            e
            for e in experiments
            if e["method"] == "picard"
            and abs(e["threshold"] - thresh) < 1e-6
        ]
        for exp in picard_exps:
            rh = exp["residual_history"]
            ax.plot(
                range(1, len(rh) + 1),
                rh,
                alpha=0.35,
                color="#42A5F5",
                linewidth=1.0,
            )
        if picard_exps:
            max_len = max(len(e["residual_history"]) for e in picard_exps)
            padded = np.full((len(picard_exps), max_len), np.nan)
            for i, e in enumerate(picard_exps):
                rh = e["residual_history"]
                padded[i, : len(rh)] = rh
            mean_curve = np.nanmean(padded, axis=0)
            ax.plot(
                range(1, max_len + 1),
                mean_curve,
                color="#1565C0",
                linewidth=2.5,
                label="Mean (base)",
            )
        ax.axhline(
            thresh,
            color="#E53935",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"threshold={thresh}",
        )
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Max Residual", fontsize=11)
        ax.set_title("Picard (base only)", fontsize=12)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # --- Two-Picard variants ---
        for col_idx, ds in enumerate(draft_steps_list):
            ax = axes[col_idx + 1]
            tp_exps = [
                e
                for e in experiments
                if e["method"] == "two_picard"
                and e["num_draft_iters"] == ds
                and abs(e["threshold"] - thresh) < 1e-6
            ]

            for exp in tp_exps:
                dr = exp.get("draft_residual_history", [])
                br = exp.get("base_residual_history", [])
                n_d = len(dr)
                if dr:
                    ax.plot(
                        range(1, n_d + 1),
                        dr,
                        alpha=0.3,
                        color="#FFD54F",
                        linewidth=1.0,
                    )
                if br:
                    ax.plot(
                        range(n_d + 1, n_d + len(br) + 1),
                        br,
                        alpha=0.3,
                        color="#42A5F5",
                        linewidth=1.0,
                    )

            # Mean curves
            if tp_exps:
                max_d = max(
                    len(e.get("draft_residual_history", [])) for e in tp_exps
                )
                max_b = max(
                    len(e.get("base_residual_history", [])) for e in tp_exps
                )
                if max_d > 0:
                    pad_d = np.full((len(tp_exps), max_d), np.nan)
                    for i, e in enumerate(tp_exps):
                        dr = e.get("draft_residual_history", [])
                        pad_d[i, : len(dr)] = dr
                    mean_d = np.nanmean(pad_d, axis=0)
                    ax.plot(
                        range(1, max_d + 1),
                        mean_d,
                        color="#F9A825",
                        linewidth=2.5,
                        label="Mean (draft 4B)",
                    )
                if max_b > 0:
                    pad_b = np.full((len(tp_exps), max_b), np.nan)
                    for i, e in enumerate(tp_exps):
                        br = e.get("base_residual_history", [])
                        pad_b[i, : len(br)] = br
                    mean_b = np.nanmean(pad_b, axis=0)
                    ax.plot(
                        range(max_d + 1, max_d + max_b + 1),
                        mean_b,
                        color="#1565C0",
                        linewidth=2.5,
                        label="Mean (base 9B)",
                    )

            ax.axhline(
                thresh,
                color="#E53935",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )
            ax.axvline(
                ds + 0.5,
                color="gray",
                linestyle=":",
                alpha=0.6,
                linewidth=1.5,
                label=f"draft→base",
            )
            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_title(f"Two-Picard (draft={ds})", fontsize=12)
            ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(
            f"Convergence Curves (threshold={thresh})", fontsize=14, y=1.02
        )
        plt.tight_layout()
        fig.savefig(
            output_dir / f"convergence_thresh_{thresh:.2f}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved: convergence_thresh_{thresh:.2f}.png")


def _plot_summary_heatmap(experiments, config, output_dir):
    """
    Heatmap: rows = methods (Picard-0d, 2P-d2, 2P-d4, 2P-d6),
    columns = prompts. Cell color = equiv sequential evals.
    One figure per threshold.
    """
    thresholds = config["thresholds"]
    draft_steps_list = config["draft_steps_list"]
    n_prompts = config["num_prompts"]

    for thresh in thresholds:
        method_names = ["Picard (base)"]
        for ds in draft_steps_list:
            method_names.append(f"2P (d={ds})")

        n_methods = len(method_names)
        evals_data = np.full((n_methods, n_prompts), np.nan)

        for exp in experiments:
            if abs(exp["threshold"] - thresh) > 1e-6:
                continue
            pi = exp["prompt_index"]
            if exp["method"] == "picard":
                evals_data[0, pi] = exp["equiv_sequential_evals"]
            elif exp["method"] == "two_picard":
                ds = exp["num_draft_iters"]
                if ds in draft_steps_list:
                    idx = 1 + draft_steps_list.index(ds)
                    evals_data[idx, pi] = exp["equiv_sequential_evals"]

        fig, ax = plt.subplots(
            figsize=(max(12, n_prompts * 0.7 + 2), n_methods * 1.0 + 2)
        )

        im = ax.imshow(
            evals_data,
            aspect="auto",
            cmap="RdYlGn_r",
            interpolation="nearest",
        )
        ax.set_yticks(range(n_methods))
        ax.set_yticklabels(method_names, fontsize=10)
        ax.set_xticks(range(n_prompts))
        ax.set_xticklabels(
            [f"P{i}" for i in range(n_prompts)], fontsize=9
        )
        ax.set_xlabel("Prompt Index", fontsize=11)
        ax.set_title(
            f"Equiv. Sequential Evals per Prompt (threshold={thresh})",
            fontsize=13,
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Equiv. Seq. Evals")

        # Annotate
        for i in range(n_methods):
            for j in range(n_prompts):
                val = evals_data[i, j]
                if not np.isnan(val):
                    vmin = np.nanmin(evals_data)
                    vmax = np.nanmax(evals_data)
                    mid = (vmin + vmax) / 2 if vmax > vmin else vmin
                    color = "white" if val > mid else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )

        plt.tight_layout()
        fig.savefig(
            output_dir / f"heatmap_thresh_{thresh:.2f}.png",
            dpi=150,
        )
        plt.close(fig)
        print(f"  Saved: heatmap_thresh_{thresh:.2f}.png")


# ──────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────


def load_transformer(model_id, dtype, device, label="transformer"):
    print(f"Loading {label} from {model_id}...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    return transformer


def load_vae(model_id, dtype, device):
    print("Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    return vae


def _experiment_key(
    method: str,
    prompt_index: int,
    threshold: float,
    num_draft_iters: int | None = None,
) -> str:
    key = f"{method}|p{prompt_index:02d}|t{threshold:.4f}"
    if num_draft_iters is not None:
        key += f"|d{num_draft_iters}"
    return key


def _metadata_path(
    metadata_dir: Path,
    method: str,
    prompt_index: int,
    threshold: float,
    num_draft_iters: int | None = None,
) -> Path:
    stem = f"{method}_p{prompt_index:02d}_t{threshold:.4f}"
    if num_draft_iters is not None:
        stem += f"_d{num_draft_iters}"
    return metadata_dir / f"{stem}.json"


def _load_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_existing_experiments(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        data = json.load(f)
    out = {}
    for exp in data.get("experiments", []):
        key = _experiment_key(
            exp["method"],
            exp["prompt_index"],
            exp["threshold"],
            exp.get("num_draft_iters"),
        )
        out[key] = exp
    return out


def _append_or_replace_experiment(
    experiments_map: dict[str, dict], exp: dict
) -> None:
    key = _experiment_key(
        exp["method"],
        exp["prompt_index"],
        exp["threshold"],
        exp.get("num_draft_iters"),
    )
    experiments_map[key] = exp


def _save_per_image_results(
    output_dir: Path,
    prompts: list[str],
    experiments: list[dict],
    num_steps: int,
):
    per_prompt = {str(i): {"prompt": p, "results": []} for i, p in enumerate(prompts)}

    for exp in experiments:
        pi = str(exp["prompt_index"])
        if exp["method"] == "euler":
            draft_iters = 0
            base_iters = num_steps
        elif exp["method"] == "picard":
            draft_iters = 0
            base_iters = int(exp.get("iters", 0))
        else:
            draft_iters = int(exp.get("draft_iters_actual", 0))
            base_iters = int(exp.get("base_iters_actual", 0))

        per_prompt[pi]["results"].append(
            {
                "method": exp["method"],
                "threshold": exp["threshold"],
                "num_draft_iters": exp.get("num_draft_iters"),
                "draft_iters_actual": draft_iters,
                "base_iters_actual": base_iters,
                "total_iters": draft_iters + base_iters,
                "equiv_sequential_evals": exp.get("equiv_sequential_evals"),
                "time_s": exp.get("time_s"),
                "final_residual": exp.get("final_residual"),
            }
        )

    _save_json(output_dir / "per_image_results.json", per_prompt)


@torch.no_grad()
def _save_intermediate_image(
    vae: AutoencoderKLFlux2 | None,
    latents: torch.Tensor,
    latent_ids: torch.Tensor,
    image_path: Path,
) -> None:
    if vae is None:
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.exists():
        return

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    lat = latents.to(device=vae_device, dtype=vae_dtype)
    l_ids = latent_ids.to(device=vae_device)
    img = decode_latent(vae, lat, l_ids)
    img.save(image_path)


# ──────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────


def run_benchmarks(
    output_dir: Path,
    prompts: list[str],
    thresholds: list[float],
    draft_steps_list: list[int],
    num_steps: int = 16,
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 4.0,
    seed: int = 42,
    dtype=torch.bfloat16,
    device=None,
    base_model_id: str = "black-forest-labs/FLUX.2-klein-9B",
    draft_model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    draft_cost_ratio: float = 0.5,
    save_images: bool = True,
    skip_euler: bool = False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    intermediate_images_dir = images_dir / "intermediate"
    intermediate_images_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    existing_experiments = _load_existing_experiments(results_path)
    experiments_map = dict(existing_experiments)

    num_train_timesteps = 1000
    vae_scale_factor = 8

    config = {
        "base_model_id": base_model_id,
        "draft_model_id": draft_model_id,
        "num_steps": num_steps,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "thresholds": thresholds,
        "draft_steps_list": draft_steps_list,
        "num_prompts": len(prompts),
        "draft_cost_ratio": draft_cost_ratio,
        "prompts": prompts,
    }

    experiments = []

    # ==================================================================
    # Text encoding: SEPARATE encoders for base (9B) and draft (4B)
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEXT ENCODING (separate encoders for 9B and 4B)")
    print("=" * 70)

    base_all_pe, base_all_ti, base_neg_pe, base_neg_ti = encode_all_prompts(
        model_id=base_model_id,
        prompts=prompts,
        device=device,
        dtype=dtype,
        label="base 9B",
    )

    draft_all_pe, draft_all_ti, draft_neg_pe, draft_neg_ti = (
        encode_all_prompts(
            model_id=draft_model_id,
            prompts=prompts,
            device=device,
            dtype=dtype,
            label="draft 4B",
        )
    )

    # ==================================================================
    # Prepare latents
    # ==================================================================
    # Load draft transformer briefly just to read in_channels config
    print("\nLoading draft transformer to read config...")
    _tmp = load_transformer(draft_model_id, dtype, device, "draft (config)")
    num_channels = _tmp.config.in_channels // 4
    del _tmp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lat_h = 2 * (height // (vae_scale_factor * 2))
    lat_w = 2 * (width // (vae_scale_factor * 2))
    raw_shape = (1, num_channels * 4, lat_h // 2, lat_w // 2)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents_raw = randn_tensor(
        raw_shape, generator=generator, device=device, dtype=dtype
    )
    latent_ids = _prepare_latent_ids(latents_raw).to(device)
    latents_init = _pack(latents_raw)
    print(f"  latents: {latents_init.shape}")

    # Build sigma schedule
    image_seq_len = latents_init.shape[1]
    mu = compute_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    sigmas = build_sigma_schedule(
        num_steps=num_steps, mu=mu, num_train_timesteps=num_train_timesteps
    )
    print(f"  sigmas: {sigmas.shape} (mu={mu:.4f})")

    vae = None
    if save_images:
        print("\nLoading VAE for intermediate/final image decoding...")
        vae = load_vae(base_model_id, dtype, device)

    # ==================================================================
    # PHASE 1: Euler baseline (base model, threshold-independent)
    # ==================================================================
    if not skip_euler:
        print("\n" + "=" * 70)
        print("PHASE 1: Euler baselines (9B base model)")
        print("=" * 70)

        base_transformer = load_transformer(
            base_model_id, dtype, device, "base transformer (Euler)"
        )

        for pi, prompt in enumerate(prompts):
            print(
                f"\n  Euler [{pi+1}/{len(prompts)}]: "
                f"\"{prompt[:50]}...\""
            )
            euler_lat_path = output_dir / f"euler_latent_p{pi:02d}.pt"
            euler_meta_path = metadata_dir / f"euler_p{pi:02d}.json"
            cached_euler = _load_json_if_exists(euler_meta_path)

            if euler_lat_path.exists() and cached_euler is not None:
                print("    cache hit: reusing existing Euler latent + metadata")
                elapsed = float(cached_euler["time_s"])
                equiv_evals = float(cached_euler["equiv_sequential_evals"])
            else:
                latents_euler = latents_init.clone()
                t0 = time.perf_counter()

                for step_i in range(num_steps):
                    latents_euler = denoise_step(
                        transformer=base_transformer,
                        latents=latents_euler,
                        latent_ids=latent_ids,
                        prompt_embeds=base_all_pe[pi],
                        text_ids=base_all_ti[pi],
                        sigma=float(sigmas[step_i]),
                        sigma_next=float(sigmas[step_i + 1]),
                        num_train_timesteps=num_train_timesteps,
                        negative_prompt_embeds=base_neg_pe,
                        negative_text_ids=base_neg_ti,
                        guidance_scale=guidance_scale,
                    )
                    _save_intermediate_image(
                        vae=vae,
                        latents=latents_euler,
                        latent_ids=latent_ids,
                        image_path=(
                            intermediate_images_dir
                            / f"euler_p{pi:02d}_s{step_i+1:03d}.png"
                        ),
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                equiv_evals = compute_equiv_sequential_evals(
                    "euler", num_steps, guidance_scale
                )

                torch.save(latents_euler.cpu(), euler_lat_path)
                _save_json(
                    euler_meta_path,
                    {
                        "method": "euler",
                        "prompt_index": pi,
                        "prompt": prompt,
                        "time_s": elapsed,
                        "equiv_sequential_evals": equiv_evals,
                        "nfe": num_steps * (2 if guidance_scale > 1 else 1),
                    },
                )
                print(
                    f"    done in {elapsed:.2f}s, "
                    f"equiv_evals={equiv_evals:.1f}"
                )

            for thresh in thresholds:
                exp = {
                    "method": "euler",
                    "prompt_index": pi,
                    "prompt": prompt,
                    "threshold": thresh,
                    "num_steps": num_steps,
                    "time_s": elapsed,
                    "equiv_sequential_evals": equiv_evals,
                    "nfe": num_steps * (2 if guidance_scale > 1 else 1),
                    "final_residual": 0.0,
                    "residual_history": [],
                }
                _append_or_replace_experiment(experiments_map, exp)

        del base_transformer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==================================================================
    # PHASE 2: Single-model Picard (9B base model)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Single-model Picard (9B base, 16 steps)")
    print("=" * 70)

    base_transformer = load_transformer(
        base_model_id, dtype, device, "base transformer (Picard)"
    )

    for thresh in thresholds:
        for pi, prompt in enumerate(prompts):
            print(
                f"\n  Picard [thresh={thresh}, prompt {pi+1}/{len(prompts)}]: "
                f"\"{prompt[:50]}...\""
            )
            final_path = output_dir / f"picard_latent_p{pi:02d}_t{thresh:.2f}.pt"
            meta_path = _metadata_path(
                metadata_dir, "picard", pi, thresh, None
            )
            cached_exp = _load_json_if_exists(meta_path)

            if final_path.exists() and cached_exp is not None:
                print("    cache hit: reusing existing Picard latent + metadata")
                exp = cached_exp
            else:
                t0 = time.perf_counter()

                latents_p, info_p = picard_trajectory(
                    transformer=base_transformer,
                    x_init=latents_init.clone(),
                    latent_ids=latent_ids,
                    prompt_embeds=base_all_pe[pi],
                    text_ids=base_all_ti[pi],
                    sigmas=sigmas,
                    num_train_timesteps=num_train_timesteps,
                    negative_prompt_embeds=base_neg_pe,
                    negative_text_ids=base_neg_ti,
                    guidance_scale=guidance_scale,
                    threshold=thresh,
                    max_picard_iters=num_steps,
                    show_progress=True,
                    save_intermediates=True,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                equiv_evals = compute_equiv_sequential_evals(
                    "picard",
                    num_steps,
                    guidance_scale,
                    iters=info_p["iters"],
                )

                exp = {
                    "method": "picard",
                    "prompt_index": pi,
                    "prompt": prompt,
                    "threshold": thresh,
                    "num_steps": num_steps,
                    "iters": info_p["iters"],
                    "time_s": elapsed,
                    "equiv_sequential_evals": equiv_evals,
                    "nfe": info_p["iters"]
                    * num_steps
                    * (2 if guidance_scale > 1 else 1),
                    "final_residual": info_p["residual"],
                    "residual_history": info_p["residual_history"],
                }

                torch.save(latents_p.cpu(), final_path)

                for iter_idx, lat in enumerate(
                    info_p.get("intermediate_latents", []), start=1
                ):
                    _save_intermediate_image(
                        vae=vae,
                        latents=lat,
                        latent_ids=latent_ids,
                        image_path=(
                            intermediate_images_dir
                            / (
                                f"picard_p{pi:02d}_t{thresh:.2f}"
                                f"_iter{iter_idx:03d}.png"
                            )
                        ),
                    )

                _save_json(meta_path, exp)
                print(
                    f"    done in {elapsed:.2f}s, iters={exp['iters']}, "
                    f"equiv_evals={equiv_evals:.1f}"
                )

            _append_or_replace_experiment(experiments_map, exp)

    del base_transformer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==================================================================
    # PHASE 3: Two-stage Picard (4B draft + 9B base)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Two-stage Picard (4B draft + 9B base)")
    print("=" * 70)

    draft_transformer = load_transformer(
        draft_model_id, dtype, device, "draft 4B (two-picard)"
    )
    base_transformer = load_transformer(
        base_model_id, dtype, device, "base 9B (two-picard)"
    )

    for thresh in thresholds:
        for ds in draft_steps_list:
            for pi, prompt in enumerate(prompts):
                print(
                    f"\n  Two-Picard [thresh={thresh}, draft_iters={ds}, "
                    f"prompt {pi+1}/{len(prompts)}]: \"{prompt[:50]}...\""
                )
                final_path = (
                    output_dir / f"two_picard_latent_p{pi:02d}_t{thresh:.2f}_d{ds}.pt"
                )
                meta_path = _metadata_path(
                    metadata_dir, "two_picard", pi, thresh, ds
                )
                cached_exp = _load_json_if_exists(meta_path)

                if final_path.exists() and cached_exp is not None:
                    print(
                        "    cache hit: reusing existing Two-Picard "
                        "latent + metadata"
                    )
                    exp = cached_exp
                else:
                    t0 = time.perf_counter()

                    latents_tp, info_tp = two_picard_trajectory(
                        base_transformer=base_transformer,
                        draft_transformer=draft_transformer,
                        x_init=latents_init.clone(),
                        latent_ids=latent_ids,
                        # Base model embeddings (9B encoder)
                        base_prompt_embeds=base_all_pe[pi],
                        base_text_ids=base_all_ti[pi],
                        base_negative_prompt_embeds=base_neg_pe,
                        base_negative_text_ids=base_neg_ti,
                        # Draft model embeddings (4B encoder)
                        draft_prompt_embeds=draft_all_pe[pi],
                        draft_text_ids=draft_all_ti[pi],
                        draft_negative_prompt_embeds=draft_neg_pe,
                        draft_negative_text_ids=draft_neg_ti,
                        # Config
                        sigmas=sigmas,
                        num_train_timesteps=num_train_timesteps,
                        guidance_scale=guidance_scale,
                        threshold=thresh,
                        draft_threshold=thresh * 2.0,
                        num_draft_iters=ds,
                        max_base_iters=num_steps,
                        show_progress=True,
                        save_intermediates=True,
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0

                    equiv_evals = compute_equiv_sequential_evals(
                        "two_picard",
                        num_steps,
                        guidance_scale,
                        draft_iters=info_tp["draft_iters"],
                        base_iters=info_tp["base_iters"],
                        draft_cost_ratio=draft_cost_ratio,
                    )

                    exp = {
                        "method": "two_picard",
                        "prompt_index": pi,
                        "prompt": prompt,
                        "threshold": thresh,
                        "num_steps": num_steps,
                        "num_draft_iters": ds,
                        "draft_iters_actual": info_tp["draft_iters"],
                        "base_iters_actual": info_tp["base_iters"],
                        "total_iters": info_tp["total_iters"],
                        "time_s": elapsed,
                        "equiv_sequential_evals": equiv_evals,
                        "nfe": (
                            info_tp["draft_iters"] * num_steps
                            + info_tp["base_iters"] * num_steps
                        )
                        * (2 if guidance_scale > 1 else 1),
                        "final_residual": info_tp["residual"],
                        "draft_residual_history": info_tp[
                            "draft_residual_history"
                        ],
                        "base_residual_history": info_tp[
                            "base_residual_history"
                        ],
                        "residual_history": (
                            info_tp["draft_residual_history"]
                            + info_tp["base_residual_history"]
                        ),
                    }

                    torch.save(latents_tp.cpu(), final_path)

                    lat_list = info_tp.get("intermediate_latents", [])
                    label_list = info_tp.get("intermediate_labels", [])
                    for iter_idx, (lat, label) in enumerate(
                        zip(lat_list, label_list), start=1
                    ):
                        _save_intermediate_image(
                            vae=vae,
                            latents=lat,
                            latent_ids=latent_ids,
                            image_path=(
                                intermediate_images_dir
                                / (
                                    f"two_picard_p{pi:02d}_t{thresh:.2f}_d{ds}"
                                    f"_iter{iter_idx:03d}_{label}.png"
                                )
                            ),
                        )

                    _save_json(meta_path, exp)
                    print(
                        f"    done in {elapsed:.2f}s, "
                        f"draft={info_tp['draft_iters']}, "
                        f"base={info_tp['base_iters']}, "
                        f"equiv_evals={equiv_evals:.1f}"
                    )

                _append_or_replace_experiment(experiments_map, exp)

    del draft_transformer, base_transformer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==================================================================
    # Save results JSON
    # ==================================================================
    experiments = list(experiments_map.values())
    results = {"config": config, "experiments": experiments}
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    _save_per_image_results(output_dir, prompts, experiments, num_steps)

    # ==================================================================
    # Decode and save images
    # ==================================================================
    if save_images and vae is not None:
        print("\nSaving final images...")

        if not skip_euler:
            for pi in range(len(prompts)):
                lat_path = output_dir / f"euler_latent_p{pi:02d}.pt"
                if lat_path.exists():
                    lat = torch.load(lat_path, weights_only=True).to(device)
                    img = decode_latent(vae, lat, latent_ids)
                    img.save(images_dir / f"euler_p{pi:02d}.png")
                    print(f"  Saved: euler_p{pi:02d}.png")

        # Decode all final outputs
        for pi in range(len(prompts)):
            for thresh in thresholds:
                lat_path = (
                    output_dir
                    / f"picard_latent_p{pi:02d}_t{thresh:.2f}.pt"
                )
                if lat_path.exists():
                    lat = torch.load(lat_path, weights_only=True).to(device)
                    img = decode_latent(vae, lat, latent_ids)
                    img.save(
                        images_dir
                        / f"picard_p{pi:02d}_t{thresh:.2f}.png"
                    )
                    print(
                        f"  Saved: picard_p{pi:02d}_t{thresh:.2f}.png"
                    )

                for ds in draft_steps_list:
                    lat_path = (
                        output_dir
                        / f"two_picard_latent_p{pi:02d}_t{thresh:.2f}_d{ds}.pt"
                    )
                    if lat_path.exists():
                        lat = torch.load(lat_path, weights_only=True).to(device)
                        img = decode_latent(vae, lat, latent_ids)
                        img.save(
                            images_dir
                            / f"two_picard_p{pi:02d}_t{thresh:.2f}_d{ds}.png"
                        )
                        print(
                            f"  Saved: two_picard_p{pi:02d}_t{thresh:.2f}_d{ds}.png"
                        )

        print(f"  Saved intermediate images to: {intermediate_images_dir}")

        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==================================================================
    # Generate plots
    # ==================================================================
    print("\nGenerating plots...")
    plot_results_from_json(results_path, plots_dir)

    # ==================================================================
    # Summary table
    # ==================================================================
    print(f"\n{'='*90}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(
        f"{'Method':<28} {'Thresh':>7} "
        f"{'Draft It':>9} {'Base It':>8} "
        f"{'Equiv Evals':>12} {'Residual':>10}"
    )
    print("-" * 90)

    for thresh in thresholds:
        # Euler
        euler_exps = [
            e
            for e in experiments
            if e["method"] == "euler"
            and abs(e["threshold"] - thresh) < 1e-6
        ]
        if euler_exps:
            mean_evals = np.mean(
                [e["equiv_sequential_evals"] for e in euler_exps]
            )
            print(
                f"{'Euler (9B)':<28} {thresh:>7.2f} "
                f"{'—':>9} {num_steps:>8} "
                f"{mean_evals:>12.1f} {'—':>10}"
            )

        # Picard (0 draft iters)
        picard_exps = [
            e
            for e in experiments
            if e["method"] == "picard"
            and abs(e["threshold"] - thresh) < 1e-6
        ]
        if picard_exps:
            mean_iters = np.mean([e["iters"] for e in picard_exps])
            mean_evals = np.mean(
                [e["equiv_sequential_evals"] for e in picard_exps]
            )
            mean_res = np.mean(
                [e["final_residual"] for e in picard_exps]
            )
            print(
                f"{'Picard (9B, d=0)':<28} {thresh:>7.2f} "
                f"{'0':>9} {mean_iters:>8.1f} "
                f"{mean_evals:>12.1f} {mean_res:>10.6f}"
            )

        # Two-Picard
        for ds in draft_steps_list:
            tp_exps = [
                e
                for e in experiments
                if e["method"] == "two_picard"
                and e["num_draft_iters"] == ds
                and abs(e["threshold"] - thresh) < 1e-6
            ]
            if tp_exps:
                mean_d = np.mean(
                    [e["draft_iters_actual"] for e in tp_exps]
                )
                mean_b = np.mean(
                    [e["base_iters_actual"] for e in tp_exps]
                )
                mean_evals = np.mean(
                    [e["equiv_sequential_evals"] for e in tp_exps]
                )
                mean_res = np.mean(
                    [e["final_residual"] for e in tp_exps]
                )
                print(
                    f"{'Two-Picard (4B+9B, d=' + str(ds) + ')':<28} "
                    f"{thresh:>7.2f} "
                    f"{mean_d:>9.1f} {mean_b:>8.1f} "
                    f"{mean_evals:>12.1f} {mean_res:>10.6f}"
                )

        print()

    print("Keeping latent files for resume/caching.")

    print(f"\nAll results in: {output_dir}/")
    print(f"  results.json  — raw experiment data")
    print(f"  per_image_results.json — per-prompt step/cost summary")
    print(f"  images/       — decoded images")
    print(f"  images/intermediate/ — decoded trajectory snapshots")
    print(f"  plots/        — comparison charts:")
    print(f"    cost_vs_draft_iters_thresh_*.png  — THE key cost comparison")
    print(f"    convergence_thresh_*.png           — residual convergence curves")
    print(f"    heatmap_thresh_*.png               — per-prompt detail heatmap")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flux2 Klein: Picard trajectory benchmarks"
    )
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Path to results.json — skip inference, just regenerate plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to test (max 15)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=16,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=256, help="Image width"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG scale",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--base-model",
        type=str,
        default="black-forest-labs/FLUX.2-klein-9B",
        help="Base (9B) model ID",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Draft (4B) model ID",
    )
    parser.add_argument(
        "--draft-cost-ratio",
        type=float,
        default=0.5,
        help="Cost ratio: draft eval / base eval (default 0.5)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.1],
        help="Picard convergence thresholds to test",
    )
    parser.add_argument(
        "--draft-steps",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Number of draft iterations to test in two-stage Picard",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image decoding/saving",
    )
    parser.add_argument(
        "--skip-euler",
        action="store_true",
        help="Skip Euler baseline (faster for debugging)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Plot-only mode ───────────────────────────────────────────────
    if args.plot_only:
        print(f"Plot-only mode: reading {args.plot_only}")
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_results_from_json(args.plot_only, plots_dir)
        print("Done!")
        exit(0)

    # ── Full benchmark mode ──────────────────────────────────────────
    prompts = BENCHMARK_PROMPTS[: args.num_prompts]

    print("=" * 70)
    print("FLUX2 KLEIN — PICARD TRAJECTORY BENCHMARK")
    print("=" * 70)
    print(f"  Base model (9B):  {args.base_model}")
    print(f"  Draft model (4B): {args.draft_model}")
    print(f"  NOTE: Each model uses its OWN text encoder.")
    print(f"  Prompts:          {len(prompts)}")
    print(f"  Steps:            {args.num_steps}")
    print(f"  Image size:       {args.width}×{args.height}")
    print(f"  CFG scale:        {args.guidance_scale}")
    print(f"  Thresholds:       {args.thresholds}")
    print(f"  Draft steps:      {args.draft_steps}")
    print(f"  Draft cost ratio: {args.draft_cost_ratio}× base")
    print(f"  Seed:             {args.seed}")
    print(f"  Output:           {output_dir}")
    print()

    n_euler = len(prompts) if not args.skip_euler else 0
    n_picard = len(prompts) * len(args.thresholds)
    n_two_picard = (
        len(prompts) * len(args.thresholds) * len(args.draft_steps)
    )
    total = n_euler + n_picard + n_two_picard
    print(
        f"  Total experiments: {total} "
        f"(euler={n_euler}, picard={n_picard}, two_picard={n_two_picard})"
    )
    print(
        f"  Text encoding: 2× (once per model) × {len(prompts)+1} prompts"
    )
    print()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16

    run_benchmarks(
        output_dir=output_dir,
        prompts=prompts,
        thresholds=args.thresholds,
        draft_steps_list=args.draft_steps,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        dtype=DTYPE,
        device=DEVICE,
        base_model_id=args.base_model,
        draft_model_id=args.draft_model,
        draft_cost_ratio=args.draft_cost_ratio,
        save_images=not args.no_images,
        skip_euler=args.skip_euler,
    )