from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

from eval_common import IMAGES_DIR, load_results, make_eval_batch, result_exists, save_result
from eval_spec_two_picard_time import MODEL_PAIRS

SPEC_NAME = "mssim"
SOURCE_SPEC_NAME = "two_picard_time"


@dataclass
class MSSIMStat:
    draft_model: str
    base_model: str
    num_steps: int
    threshold: float
    draft_init: int
    draft_iters: float
    base_iters: float
    mssim_score: float


image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.PILToTensor(),
    ]
)


def _build_real_image_bank(ds: Dataset, labels: list[int]) -> dict[int, list[torch.Tensor]]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    label_to_indices: dict[int, list[int]] = {}
    for idx, label in tqdm(enumerate(ds["label"]), total=len(ds), desc="index dataset", leave=False):
        if label in counts:
            label_to_indices.setdefault(int(label), []).append(idx)

    real_bank: dict[int, list[torch.Tensor]] = {}
    for label, needed in tqdm(counts.items(), desc="load reference images", leave=False):
        selected = ds.select(label_to_indices[label])  # type: ignore
        shuffled = selected.shuffle(seed=42).select(range(needed)) # type: ignore

        def transform_batch(batch: dict[str, Any]) -> dict[str, Any]:
            batch["image_tensor"] = [image_transform(img) for img in batch["image"]]
            return batch

        transformed = shuffled.with_transform(transform_batch)
        real_bank[label] = [transformed[i]["image_tensor"] for i in range(needed)]

    return real_bank


def run(force: bool = False) -> None:
    raw_results = load_results(SOURCE_SPEC_NAME)
    if not raw_results:
        print("No two_picard_time results found.")
        return

    valid_prefixes = {
        f"two_picard_draft_{pair['draft']}_base_{pair['base']}_steps_{num_steps}_threshold_{threshold}_draft_init_"
        for pair in MODEL_PAIRS
        for num_steps in pair["num_steps"]
        for threshold in pair["thresholds"]
    }

    eval_items = [
        (key, records)
        for key, records in raw_results.items()
        if any(key.startswith(prefix) for prefix in valid_prefixes)
    ]
    if not eval_items:
        print("No matching two_picard_time configs found for MSSIM evaluation.")
        return

    num_images = max(len(records) for _, records in eval_items if records)
    _, y, _ = make_eval_batch(num_images)
    labels = [int(label.item()) for label in y]

    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    real_bank = _build_real_image_bank(ds, labels)
    metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    for eval_key, records in tqdm(eval_items, desc="mssim configs"):
        if not records:
            continue
        if not force and result_exists(SPEC_NAME, eval_key):
            continue

        image_dir = IMAGES_DIR / SOURCE_SPEC_NAME / eval_key
        if not image_dir.exists():
            print(f"Skipping {eval_key}: missing image directory {image_dir}")
            continue

        real_offsets = {label: 0 for label in real_bank}
        real_images: list[torch.Tensor] = []
        gen_images: list[torch.Tensor] = []

        for record in records:
            img_idx = int(record["img_idx"])
            label = labels[img_idx]
            offset = real_offsets[label]
            if offset >= len(real_bank[label]):
                raise RuntimeError(f"Insufficient reference images for label {label}")

            real_images.append(real_bank[label][offset].float() / 255.0)
            real_offsets[label] += 1

            image_path = image_dir / f"img_{img_idx:03d}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing generated image: {image_path}")
            gen_images.append(read_image(str(image_path)).float() / 255.0)

        real_batch = torch.stack(real_images, dim=0)
        gen_batch = torch.stack(gen_images, dim=0)
        mssim_score = float(metric(gen_batch, real_batch).item())
        metric.reset()

        save_result(
            SPEC_NAME,
            eval_key,
            [
                asdict(
                    MSSIMStat(
                        draft_model=str(records[0]["draft_model"]),
                        base_model=str(records[0]["base_model"]),
                        num_steps=int(records[0]["num_steps"]),
                        threshold=float(records[0]["threshold"]),
                        draft_init=int(records[0]["draft_init"]),
                        draft_iters=float(sum(float(record["draft_iters"]) for record in records) / len(records)),
                        base_iters=float(sum(float(record["base_iters"]) for record in records) / len(records)),
                        mssim_score=mssim_score,
                    )
                )
            ],
        )


if __name__ == "__main__":
    with torch.no_grad():
        run()
