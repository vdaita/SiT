import argparse
import torch
from models import SiT_XL_2
from download import find_model
from inference import speculative_trajectory, picard_trajectory, two_picard_trajectory

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
NUM_STEPS = 32
cfg_scale = 4.0

THRESHOLD = 0.01
batch_size = 1

torch.manual_seed(SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, default="models/XL.pt")
    parser.add_argument(
        "--draft-ckpt",
        type=str,
        default="checkpoints/distill/wandering-mountain-8/draft_model_final.pt",
    )
    args = parser.parse_args()

    with torch.no_grad():
        base_model = SiT_XL_2(input_size=LATENT_SIZE).to(DEVICE)
        base_model.load_state_dict(find_model(args.base_ckpt))
        base_model.eval()

        draft_model = SiT_S_2_Projected(input_size=LATENT_SIZE).to(DEVICE)
        draft_model.load_state_dict(
            find_model(args.draft_ckpt),
            strict=False,
        )
        draft_model.eval()

        x_in = torch.randn((batch_size, 4, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
        y_null = torch.full((batch_size,), 1000, device=DEVICE)

        picard_images, picard_stats = picard_trajectory(base_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
        print("Picard iteration states: ", picard_stats)

        two_picard_images, two_picard_stats = two_picard_trajectory(base_model, draft_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
        print("Two picard iteration states: ", two_picard_stats)