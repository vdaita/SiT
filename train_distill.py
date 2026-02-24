"""
Distill latents from the largest model to the smallest model.
"""
from argparse import Namespace
import argparse
import wandb
from models import SiT_XL_2, SiT_S_2, SiT
from download import find_model
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
NUM_STEPS = 32
cfg_scale = 4.0

def save_model(model: SiT, iter_num: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    filename = f"draft_model_{iter_num}.pt"
    path = os.path.join(folder, filename)
    torch.save(model.state_dict(), path)

def main(args: Namespace):
    base_model = SiT_XL_2(input_size=LATENT_SIZE).to(DEVICE)
    base_model.load_state_dict(find_model("models/XL.pt"))
    base_model.eval()

    draft_model = SiT_S_2(input_size=LATENT_SIZE).to(DEVICE)
    draft_model.load_state_dict(find_model("models/S.pt"))
    draft_model.train()

    optimizer = torch.optim.Adam(draft_model.parameters(), lr=args.lr)

    for train_step in tqdm(range(args.num_train_steps)):
        y = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=DEVICE)
        x = torch.randn(args.batch_size, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
        dt = 1.0 / NUM_STEPS
        y_null = torch.full((args.batch_size,), 1000, device=DEVICE)
        optimizer.zero_grad()

        total_loss = 0.0

        for diffusion_step in range(NUM_STEPS):
            x_model = torch.cat([x, x], dim=0)
            t = torch.full((args.batch_size, ), 1 / NUM_STEPS, device=DEVICE)
            t_model = torch.cat([t, t], dim=0)
            y_model = torch.cat([y, y_null], dim=0)


            # forward pass the base model
            with torch.no_grad():
                v_base_model = base_model(x_model, t_model, y_model)
                v_base_uncond, v_base_cond = torch.chunk(v_base_model, 2, dim=0)
                v_base = v_base_uncond + cfg_scale * (v_base_cond - v_base_uncond)

            # forward pass the draft model
            with torch.enable_grad():
                v_draft_model = draft_model(x_model, t_model, y_model)
                v_draft_uncond, v_draft_cond = torch.chunk(v_draft_model, 2, dim=0)
                v_draft = v_draft_uncond + cfg_scale * (v_draft_cond - v_draft_uncond)
            
            x = x + dt * v_base
            loss = F.mse_loss(v_base, v_draft) / NUM_STEPS
            total_loss += loss.item()
            loss.backward()

            if train_step * NUM_STEPS + diffusion_step == args.checkpoint_every:
                save_model(draft_model, f"{train_step}-{diffusion_step}", args.checkpoint_dir)

        # compute the loss
        optimizer.step()
        wandb.log({"loss": total_loss, "step": train_step})

    save_model(draft_model, "final", args.checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/distill")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    wandb.init(project="sit-distill")
    main(args)