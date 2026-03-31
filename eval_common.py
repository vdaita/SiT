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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
CFG_SCALE = 4.0

models = {
    "S": SiT_S_2().to(DEVICE),
    "B": SiT_B_2().to(DEVICE),
    "L": SiT_L_2().to(DEVICE)
}

def get_vae():
    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE)

def vae_decode(vae, latent):
    return vae.decode(latent / 0.18215).sample