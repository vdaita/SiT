from typing import TypedDict, Dict, Tuple, List
from tqdm import tqdm
from eval_common import models, DEVICE, IMAGE_SIZE, LATENT_SIZE, NUM_CLASSES, CFG_SCALE, get_vae, vae_decode
from inference import two_picard_trajectory_grid
from datasets import load_dataset, Dataset
import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms

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

BATCH_SIZE = 1

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

two_picard_grid_configs: List[TwoPicardGridConfig] = [
    {
        "draft": "S",
        "base": "B",
        "num_steps": 32,
        "num_classes": 32,
        "images_per_class": 32,
        "pairs": [
            {
                "draft_iters": 8,
                "base_iters": 2
            },
            {
                "draft_iters": 8,
                "base_iters": 4
            },
            {
                "draft_iters": 8,
                "base_iters": 6
            },
            {
                "draft_iters": 8,
                "base_iters": 8
            },
            {
                "draft_iters": 4,
                "base_iters": 2
            },
            {
                "draft_iters": 4,
                "base_iters": 4,
            },
            {
                "draft_iters": 4,
                "base_iters": 6
            },
            {
                "draft_iters": 4,
                "base_iters": 8
            }
        ]
    },
    {
        "draft": "S",
        "base": "L",
        "num_steps": 32,
        "num_classes": 32,
        "images_per_class": 32,
        "pairs": [
            {
                "draft_iters": 8,
                "base_iters": 2
            },
            {
                "draft_iters": 8,
                "base_iters": 4
            },
            {
                "draft_iters": 8,
                "base_iters": 6
            },
            {
                "draft_iters": 8,
                "base_iters": 8
            },
            {
                "draft_iters": 4,
                "base_iters": 2
            },
            {
                "draft_iters": 4,
                "base_iters": 4,
            },
            {
                "draft_iters": 4,
                "base_iters": 6
            },
            {
                "draft_iters": 4,
                "base_iters": 8
            }
        ]
    }
]

if __name__ == "__main__":
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation")
    vae = get_vae()

    for model_size_pair in tqdm(two_picard_grid_configs, desc="grid configurations model size"):
        base_id, draft_id, num_steps = model_size_pair["base"], model_size_pair["draft"], model_size_pair["num_steps"]
        base_model = models[model_size_pair["base"]]
        draft_model = models[model_size_pair["draft"]]
        pairs = [(pair["draft_iters"], pair["base_iters"]) for pair in model_size_pair["pairs"]]
        
        num_images, num_classes = model_size_pair["num_classes"], model_size_pair["images_per_class"]
        kid_scores = {pair: [] for pair in pairs}

        for class_idx in tqdm(range(num_images), desc="classes for two picard", leave=True):
            selected_images = ds.filter(lambda row: (row["label"] == class_idx)) # type: Dataset

            class_x = [torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)]
            class_y = [torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)]
            class_y_null = [torch.full((BATCH_SIZE, ), NUM_CLASSES, device=DEVICE)]

            real_images = selected_images.shuffle(seed=42).select(range(num_images)) # type: ignore
            def transform_batch(batch):
                batch["image_tensor"] = [image_transform(img) for img in batch["image"]]
                return batch
            real_images = real_images.with_transform(transform_batch)
            real_images = [real_images[i]["image_tensor"] for i in range(num_images)]
            real_images = torch.stack(real_images, dim=0)

            gen_images_list: Dict[Tuple[int, int], List] = {}

            metric = KernelInceptionDistance()
            metric.update(real_images, real=True)

            eval_key = f"two_picard_base_{base_model}_draft_{draft_model}_steps_{num_steps}_class_{class_idx}"

            if exists_result(eval_key):
                load_results(eval_key)
                continue

            # if these results don't exist...
            for idx in tqdm(range(num_images), desc=f"two picard base={base_model} draft={draft_model} class={class_idx}", leave=True):
               
                pair_outputs = two_picard_trajectory_grid(
                    base_model,
                    draft_model,
                    class_x[idx],
                    class_y[idx],
                    class_y_null[idx],
                    num_steps,
                    pairs,
                    CFG_SCALE
                )
                
                for pair in pairs:
                    decoded_image = vae_decode(vae, pair_outputs[pair])
                    gen_images_list[pair].append(decoded_image)
            
            for pair in pairs:
                fake_stack = torch.stack(gen_images_list[pair])
                metric.update(fake_stack, real=False)
                kid_score = metric.compute()[0]
                kid_scores[pair].append(kid_score)

            save_result(...) # TODO: save the KID score
        