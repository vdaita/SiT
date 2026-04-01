from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from eval_common import IMAGES_DIR, load_results, result_exists, save_result
from eval_spec_two_picard_grid import two_picard_grid_configs

SPEC_NAME = "mssim"
SOURCE_SPEC_NAME = "two_picard_grid"


@dataclass
class MSSIMStat:
    draft_model: str
    base_model: str
    num_steps: int
    class_idx: int
    draft_iters: int
    base_iters: int
    mssim_score: float


image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.PILToTensor(),
    ]
)


def _transform_batch(batch: dict[str, Any]) -> dict[str, Any]:
    batch["image_tensor"] = [image_transform(img) for img in batch["image"]]
    return batch


def run(force: bool = False) -> None:
    raw_results = load_results(SOURCE_SPEC_NAME)
    if not raw_results:
        print("No two_picard_grid results found.")
        return

    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in tqdm(enumerate(ds["label"]), total=len(ds), desc="index dataset"):
        label_to_indices[int(label)].append(idx)

    metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    selected_classes = [1, 32, 64, 128, 256, 512, 590, 780]

    for model_size_pair in tqdm(two_picard_grid_configs, desc="mssim model pairs"):
        draft_id = model_size_pair["draft"]
        base_id = model_size_pair["base"]
        num_steps = model_size_pair["num_steps"]
        num_classes = model_size_pair["num_classes"]
        images_per_class = model_size_pair["images_per_class"]
        pairs = [(pair["draft_iters"], pair["base_iters"]) for pair in model_size_pair["pairs"]]

        for class_idx in tqdm(selected_classes[:num_classes], desc="mssim classes", leave=False):
            eval_key = (
                f"two_picard_grid_draft_{draft_id}_base_{base_id}"
                f"_steps_{num_steps}_class_{class_idx}"
            )
            if eval_key not in raw_results:
                continue
            if not force and result_exists(SPEC_NAME, eval_key):
                continue

            selected_images = ds.select(label_to_indices[class_idx])  # type: ignore[arg-type]
            real_images_ds = selected_images.shuffle(seed=42).select(range(images_per_class))
            real_images_ds = real_images_ds.with_transform(_transform_batch)
            real_images = [real_images_ds[i]["image_tensor"].float() / 255.0 for i in range(images_per_class)]
            real_batch = torch.stack(real_images, dim=0)

            pair_stats = []
            for draft_iters, base_iters in pairs:
                image_dir = IMAGES_DIR / SOURCE_SPEC_NAME / f"{eval_key}_draft_{draft_iters}_base_{base_iters}"
                if not image_dir.exists():
                    raise FileNotFoundError(f"Missing generated image directory: {image_dir}")

                gen_images = []
                for img_idx in range(images_per_class):
                    image_path = image_dir / f"img_{img_idx:03d}.png"
                    if not image_path.exists():
                        raise FileNotFoundError(f"Missing generated image: {image_path}")
                    gen_images.append(read_image(str(image_path)).float() / 255.0)

                gen_batch = torch.stack(gen_images, dim=0)
                mssim_score = float(metric(gen_batch, real_batch).item())
                metric.reset()

                pair_stats.append(
                    asdict(
                        MSSIMStat(
                            draft_model=draft_id,
                            base_model=base_id,
                            num_steps=num_steps,
                            class_idx=class_idx,
                            draft_iters=draft_iters,
                            base_iters=base_iters,
                            mssim_score=mssim_score,
                        )
                    )
                )

            save_result(SPEC_NAME, eval_key, pair_stats)


if __name__ == "__main__":
    with torch.no_grad():
        run()
