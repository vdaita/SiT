import argparse
import os
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models import SiT_XL_2, SiT_S_2, SiT_B_2
from download import find_model
from inference import speculative_trajectory, picard_trajectory, two_picard_trajectory, straight_line_speculation

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
NUM_STEPS = 32
cfg_scale = 4.0
NUM_DRAFT_STEPS = 4

THRESHOLD = 0.005
batch_size = 1
OUTPUT_DIR = "outputs"

torch.manual_seed(SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=str, default="models/XL.pt")
    # parser.add_argument("--draft-ckpt", type=str, default="models/S.pt")
    parser.add_argument("--draft-ckpt", type=str, default="models/B.pt")
    parser.add_argument("--draft-ckpt-trained", type=str, required=False)
    args = parser.parse_args()

    with torch.no_grad():
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        base_model = SiT_XL_2(input_size=LATENT_SIZE).to(DEVICE)
        base_model.load_state_dict(find_model(args.base_ckpt))
        base_model.eval()

        draft_model = SiT_B_2(input_size=LATENT_SIZE).to(DEVICE)
        draft_model.load_state_dict(find_model(args.draft_ckpt))
        draft_model.eval()

        # draft_model = SiT_S_2(input_size=LATENT_SIZE).to(DEVICE)
        # draft_model.load_state_dict(
        #     find_model(args.draft_ckpt)
        # )
        # draft_model.eval()

        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(DEVICE)

        x_in = torch.randn((batch_size, 4, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
        y_null = torch.full((batch_size,), 1000, device=DEVICE)

        # picard_images, picard_stats = picard_trajectory(base_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
        # print("Picard iteration states: ", picard_stats)
        # picard_decoded = vae.decode(picard_images / 0.18215).sample
        # save_image(picard_decoded, os.path.join(OUTPUT_DIR, "picard.png"), nrow=1, normalize=True, value_range=(-1, 1))

        # two_picard_images, two_picard_stats = two_picard_trajectory(base_model, draft_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
        # print("Two picard iteration states: ", two_picard_stats)
        # two_picard_decoded = vae.decode(two_picard_images / 0.18215).sample
        # save_image(two_picard_decoded, os.path.join(OUTPUT_DIR, "two_picard.png"), nrow=1, normalize=True, value_range=(-1, 1))

        straight_spec_images, straight_spec_stats = straight_line_speculation(
            base_model,
            x_in,
            y,
            y_null,
            NUM_STEPS,
            NUM_DRAFT_STEPS,
            cfg_scale,
            THRESHOLD,
            show_progress=True,
        )
        print("Speculative iteration states: ", straight_spec_stats)
        speculative_decoded = vae.decode(straight_spec_images / 0.18215).sample
        save_image(speculative_decoded, os.path.join(OUTPUT_DIR, "straight_spec.png"), nrow=1, normalize=True, value_range=(-1, 1))


        # speculative_images, speculative_stats = speculative_trajectory(
        #     base_model,
        #     draft_model,
        #     x_in,
        #     y,
        #     y_null,
        #     NUM_STEPS,
        #     NUM_DRAFT_STEPS,
        #     cfg_scale,
        #     THRESHOLD,
        #     show_progress=True,
        # )
        # print("Speculative iteration states: ", speculative_stats)
        # speculative_decoded = vae.decode(speculative_images / 0.18215).sample
        # save_image(speculative_decoded, os.path.join(OUTPUT_DIR, "speculative.png"), nrow=1, normalize=True, value_range=(-1, 1))
