import argparse
import os
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models import SiT_XL_2, SiT_S_2, SiT_B_2, SiT_B_2_short, SiT_S_2_short
from download import find_model
from inference import speculative_trajectory, picard_trajectory, two_picard_trajectory, straight_line_speculation, multi_stage_trajectory, parareal, sequential_trajectory

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
    parser.add_argument("--num-iters", type=int, default=1)
    args = parser.parse_args()

    for iter in range(args.num_iters):
        with torch.no_grad():
            iter_out = f"{OUTPUT_DIR}/{iter}"
            os.makedirs(iter_out, exist_ok=True)

            base_model = SiT_B_2(input_size=LATENT_SIZE).to(DEVICE)
            base_model.load_state_dict(find_model("models/B.pt"))
            base_model.eval()

            draft_model = SiT_B_2_short(input_size=LATENT_SIZE).to(DEVICE)
            draft_model.load_state_dict(find_model("checkpoints/distill/comic-pond-39/draft_model_final.pt"))
            draft_model.eval()

            # draft_model = SiT_S_2(input_size=LATENT_SIZE).to(DEVICE)
            # draft_model.load_state_dict(
            #     find_model("models/S.pt")
            # )
            # draft_model.eval()

            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(DEVICE)

            x_in = torch.randn((batch_size, 4, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
            y = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
            y_null = torch.full((batch_size,), 1000, device=DEVICE)

            picard_images, picard_stats = picard_trajectory(base_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
            print("Picard iteration states: ", picard_stats)
            picard_decoded = vae.decode(picard_images / 0.18215).sample
            save_image(picard_decoded, os.path.join(iter_out, "picard.png"), nrow=1, normalize=True, value_range=(-1, 1))

            picard_draft_images, picard_draft_stats = picard_trajectory(draft_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD / 20, show_progress=True)
            print("Picard draft iteration states: ", picard_draft_stats)
            picard_draft_decoded = vae.decode(picard_draft_images / 0.18215).sample
            save_image(picard_draft_decoded, os.path.join(iter_out, "picard_draft.png"), nrow=1, normalize=True, value_range=(-1, 1))

            sequential_draft_images, sequential_draft_stats = sequential_trajectory(draft_model, x_in, y, y_null, NUM_STEPS, cfg_scale, show_progress=True)
            print("Speculative draft iteration states: ", picard_draft_stats)
            sequential_draft_decoded = vae.decode(sequential_draft_images / 0.18215).sample
            save_image(sequential_draft_decoded, os.path.join(iter_out, "sequential_draft.png"), nrow=1, normalize=True, value_range=(-1, 1))

            two_picard_images, two_picard_stats = two_picard_trajectory(base_model, draft_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
            print("Two picard iteration states: ", two_picard_stats)
            two_picard_decoded = vae.decode(two_picard_images / 0.18215).sample
            save_image(two_picard_decoded, os.path.join(OUTPUT_DIR, "two_picard.png"), nrow=1, normalize=True, value_range=(-1, 1))

            # multi_stage_images, multi_stage_stats = multi_stage_trajectory(base_model, x_in, y, y_null, 8, 4, cfg_scale, THRESHOLD, show_progress=True)
            # print("Multi stage stats: ", multi_stage_stats)
            # multi_stage_decoded = vae.decode(multi_stage_images / 0.18215).sample
            # save_image(multi_stage_decoded, os.path.join(iter_out, "multi_stage.png"), nrow=1, normalize=True, value_range=(-1, 1))

            # parareal_images, parareal_stats = parareal(base_model, x_in, y, y_null, 8, 4, cfg_scale, THRESHOLD, show_progress=True)
            # print("Parareal stats: ", parareal_stats)
            # parareal_decoded = vae.decode(parareal_images / 0.18215).sample
            # save_image(parareal_decoded, os.path.join(iter_out, "parareal.png"), nrow=1, normalize=True, value_range=(-1, 1))


        # straight_spec_images, straight_spec_stats = straight_line_speculation(
        #     base_model,
        #     x_in,
        #     y,
        #     y_null,
        #     NUM_STEPS,
        #     NUM_DRAFT_STEPS,
        #     cfg_scale,
        #     THRESHOLD,
        #     show_progress=True,
        # )
        # print("Speculative iteration stats: ", straight_spec_stats)
        # speculative_decoded = vae.decode(straight_spec_images / 0.18215).sample
        # for rhi in range(straight_spec_stats["iters"]):
        #     print(f"Iteration {rhi}:", straight_spec_stats["residual_history"][rhi])

        # save_image(speculative_decoded, os.path.join(OUTPUT_DIR, "straight_spec.png"), nrow=1, normalize=True, value_range=(-1, 1))

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
            # save_image(speculative_decoded, os.path.join(iter_out, "speculative.png"), nrow=1, normalize=True, value_range=(-1, 1))
