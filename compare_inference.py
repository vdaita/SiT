import torch
from tqdm import tqdm
from models import SiT_XL_2, SiT_S_2, SiT
from download import find_model
from inference import speculative_trajectory, picard_trajectory

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
NUM_STEPS = 32
cfg_scale = 4.0

THRESHOLD = 0.01
NUM_DRAFT_STEPS = 4
batch_size = 1

torch.manual_seed(SEED)

if __name__ == "__main__":
    with torch.no_grad():
        base_model = SiT_XL_2(input_size=LATENT_SIZE).to(DEVICE)
        base_model.load_state_dict(find_model("models/XL.pt"))
        base_model.eval()

        draft_model = SiT_S_2(input_size=LATENT_SIZE).to(DEVICE)
        draft_model.load_state_dict(find_model("checkpoints/distill/wandering-mountain-8/draft_model_final.pt"))
        draft_model.eval()

        x_in = torch.randn((batch_size, 4, LATENT_SIZE, LATENT_SIZE), device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (batch_size,), device=DEVICE)
        y_null = torch.full((batch_size,), 1000, device=DEVICE)

        picard_images, picard_stats = picard_trajectory(base_model, x_in, y, y_null, NUM_STEPS, cfg_scale, THRESHOLD, show_progress=True)
        print("Picard iteration states: ", picard_stats)

        speculative_images, speculative_stats = speculative_trajectory(base_model, draft_model, x_in, y, y_null, NUM_STEPS, NUM_DRAFT_STEPS, cfg_scale, THRESHOLD, show_progress=True)
        print("Speculative iteration stats: ", speculative_stats)
        
