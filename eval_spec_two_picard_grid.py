from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, TypedDict

import torch
from datasets import Dataset, load_dataset
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms
from tqdm import tqdm

from eval_common import (
    CFG_SCALE,
    DEVICE,
    LATENT_SIZE,
    NUM_CLASSES,
    get_available_models,
    get_vae,
    load_model,
    result_exists,
    save_decoded_image,
    save_result,
)
from inference import two_picard_trajectory_grid

SPEC_NAME = "two_picard_grid"
BATCH_SIZE = 1


class TwoPicardGridConfigPair(TypedDict):
    draft_iters: int
    base_iters: int


class TwoPicardGridConfig(TypedDict):
    draft: str
    base: str
    num_steps: int
    num_classes: int
    images_per_class: int
    pairs: List[TwoPicardGridConfigPair]


@dataclass
class TwoPicardGridStat:
    draft_iters: int
    base_iters: int
    kid: float


image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)

two_picard_grid_configs: List[TwoPicardGridConfig] = [
    {
        "draft": "S",
        "base": "B",
        "num_steps": 32,
        "num_classes": 32,
        "images_per_class": 32,
        "pairs": [
            {"draft_iters": 8, "base_iters": 2},
            {"draft_iters": 8, "base_iters": 4},
            {"draft_iters": 8, "base_iters": 6},
            {"draft_iters": 8, "base_iters": 8},
            {"draft_iters": 4, "base_iters": 2},
            {"draft_iters": 4, "base_iters": 4},
            {"draft_iters": 4, "base_iters": 6},
            {"draft_iters": 4, "base_iters": 8},
        ],
    },
    {
        "draft": "S",
        "base": "L",
        "num_steps": 32,
        "num_classes": 32,
        "images_per_class": 32,
        "pairs": [
            {"draft_iters": 8, "base_iters": 2},
            {"draft_iters": 8, "base_iters": 4},
            {"draft_iters": 8, "base_iters": 6},
            {"draft_iters": 8, "base_iters": 8},
            {"draft_iters": 4, "base_iters": 2},
            {"draft_iters": 4, "base_iters": 4},
            {"draft_iters": 4, "base_iters": 6},
            {"draft_iters": 4, "base_iters": 8},
        ],
    },
]


def run(force: bool = False) -> None:
    available = set(get_available_models())
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation")
    vae = get_vae()

    for model_size_pair in tqdm(two_picard_grid_configs, desc="grid model pairs"):
        if model_size_pair["draft"] not in available or model_size_pair["base"] not in available:
            continue

        base_id = model_size_pair["base"]
        draft_id = model_size_pair["draft"]
        num_steps = model_size_pair["num_steps"]
        num_classes = model_size_pair["num_classes"]
        images_per_class = model_size_pair["images_per_class"]
        base_model = load_model(base_id)
        draft_model = load_model(draft_id)
        pairs = [(pair["draft_iters"], pair["base_iters"]) for pair in model_size_pair["pairs"]]

        for class_idx in tqdm(range(num_classes), desc="grid classes", leave=False):
            eval_key = (
                f"two_picard_grid_draft_{draft_id}_base_{base_id}"
                f"_steps_{num_steps}_class_{class_idx}"
            )
            if not force and result_exists(SPEC_NAME, eval_key):
                continue

            selected_images = ds.filter(lambda row: row["label"] == class_idx)  # type: Dataset
            real_images = selected_images.shuffle(seed=42).select(range(images_per_class))

            def transform_batch(batch):
                batch["image_tensor"] = [image_transform(img) for img in batch["image"]]
                return batch

            real_images = real_images.with_transform(transform_batch)
            real_images = [real_images[i]["image_tensor"] for i in range(images_per_class)]
            real_images = torch.stack(real_images, dim=0)

            class_x = [
                torch.randn(BATCH_SIZE, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
                for _ in range(images_per_class)
            ]
            class_y = [torch.full((BATCH_SIZE,), class_idx, device=DEVICE) for _ in range(images_per_class)]
            class_y_null = [torch.full((BATCH_SIZE,), NUM_CLASSES, device=DEVICE) for _ in range(images_per_class)]

            gen_images_list = {pair: [] for pair in pairs}

            for idx in tqdm(range(images_per_class), desc=eval_key, leave=False):
                pair_outputs = two_picard_trajectory_grid(
                    base_model,
                    draft_model,
                    class_x[idx],
                    class_y[idx],
                    class_y_null[idx],
                    num_steps,
                    pairs,
                    CFG_SCALE,
                )

                for pair in pairs:
                    decoded_image = vae.decode(pair_outputs[pair] / 0.18215).sample
                    gen_images_list[pair].append(decoded_image.squeeze(0).cpu())
                    save_decoded_image(
                        SPEC_NAME,
                        f"{eval_key}_draft_{pair[0]}_base_{pair[1]}",
                        idx,
                        decoded_image,
                    )

            pair_stats = []
            for pair in pairs:
                metric = KernelInceptionDistance(subset_size=min(50, images_per_class))
                metric.update(real_images, real=True)
                metric.update(torch.stack(gen_images_list[pair], dim=0), real=False)
                kid_score = float(metric.compute()[0])
                pair_stats.append(
                    asdict(
                        TwoPicardGridStat(
                            draft_iters=pair[0],
                            base_iters=pair[1],
                            kid=kid_score,
                        )
                    )
                )

            save_result(SPEC_NAME, eval_key, pair_stats)


if __name__ == "__main__":
    run()
