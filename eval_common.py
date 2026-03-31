from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from torchvision.io import read_image

from download import find_model
from models import SiT_B_2, SiT_L_2, SiT_S_2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
CFG_SCALE = 4.0
BATCH_SIZE = 1
SEED = 42
NUM_IMAGES = 32

OUTPUTS_DIR = Path("outputs")
RESULTS_DIR = OUTPUTS_DIR / "spec_results"
IMAGES_DIR = OUTPUTS_DIR / "spec_images"

MODEL_REGISTRY = {
    "S": (SiT_S_2, "models/S.pt"),
    "B": (SiT_B_2, "models/B.pt"),
    "L": (SiT_L_2, "models/L.pt"),
}

_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_VAE: AutoencoderKL | None = None


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def result_path(spec_name: str) -> Path:
    return RESULTS_DIR / f"{spec_name}.json"


def load_results(spec_name: str) -> dict[str, Any]:
    path = result_path(spec_name)
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_results(spec_name: str, data: dict[str, Any]) -> None:
    ensure_output_dirs()
    path = result_path(spec_name)
    with path.open("w") as f:
        json.dump(_to_jsonable(data), f, indent=2)


def result_exists(spec_name: str, key: str) -> bool:
    return key in load_results(spec_name)


def save_result(spec_name: str, key: str, value: Any) -> None:
    data = load_results(spec_name)
    data[key] = _to_jsonable(value)
    save_results(spec_name, data)


def load_model(model_name: str) -> torch.nn.Module:
    if model_name not in _MODEL_CACHE:
        model_cls, ckpt_path = MODEL_REGISTRY[model_name]
        model = model_cls(input_size=LATENT_SIZE).to(DEVICE)
        model.load_state_dict(find_model(ckpt_path), strict=True)
        model.eval()
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def get_available_models() -> list[str]:
    return [name for name, (_, ckpt_path) in MODEL_REGISTRY.items() if os.path.exists(ckpt_path)]


def get_vae() -> AutoencoderKL:
    global _VAE
    if _VAE is None:
        _VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)
    return _VAE


def vae_decode(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    return vae.decode(latent / 0.18215).sample


def save_decoded_image(spec_name: str, eval_key: str, image_idx: int, decoded: torch.Tensor) -> torch.Tensor:
    ensure_output_dirs()
    image_dir = IMAGES_DIR / spec_name / eval_key
    image_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        decoded,
        image_dir / f"img_{image_idx:03d}.png",
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    
    return read_image(f"{image_dir}/img_{image_idx:03d}.png")


def images_complete(spec_name: str, eval_key: str, num_images: int) -> bool:
    image_dir = IMAGES_DIR / spec_name / eval_key
    if not image_dir.exists():
        return False
    return len(list(image_dir.glob("img_*.png"))) >= num_images


def make_eval_batch(num_images: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    torch.manual_seed(SEED)
    x = [torch.randn(BATCH_SIZE, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE) for _ in range(num_images)]
    y = [torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE) for _ in range(num_images)]
    y_null = [torch.full((BATCH_SIZE,), NUM_CLASSES, device=DEVICE) for _ in range(num_images)]
    return x, y, y_null
