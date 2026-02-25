"""
Distill latents from the largest model to the smallest model.
"""
from argparse import Namespace
import argparse
import wandb
from models import SiT_XL_2, SiT_S_2_Projected, SiT
from download import find_model
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 1000
NUM_STEPS = 16
cfg_scale = 4.0

torch.manual_seed(SEED)

def save_model(model: SiT, iter_num: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    filename = f"draft_model_{iter_num}.pt"
    path = os.path.join(folder, filename)
    torch.save(model.state_dict(), path)

def main(args: Namespace):

    base_model = SiT_XL_2(input_size=LATENT_SIZE).to(DEVICE)
    base_model.load_state_dict(find_model("models/XL.pt"))
    base_model.eval()
    base_model.requires_grad_(False)

    draft_model = SiT_S_2_Projected(input_size=LATENT_SIZE).to(DEVICE)
    draft_model.load_state_dict(find_model("models/S.pt"), strict=False)
    draft_model.train()
    draft_model.requires_grad_(True)

    optimizer = torch.optim.Adam(draft_model.parameters(), lr=args.lr)

    for train_step in tqdm(range(args.num_train_steps)):
        optimizer.zero_grad()

        z = torch.randn(args.batch_size, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=DEVICE)
        y_null = torch.full((args.batch_size,), 1000, device=DEVICE)
        t = torch.rand(args.batch_size, device=DEVICE)

        dt = 1 / NUM_STEPS

        with torch.no_grad():
            x_in = torch.cat([z, z], dim=0)
            t_in = torch.cat([t, t], dim=0)
            y_in = torch.cat([y, y_null], dim=0)

            v_base_flat, (v_base_hidden, v_base_emb) = base_model(x_in, t_in, y_in, return_hidden=True)
            
            v_base_cond, v_base_uncond = v_base_flat.chunk(2, dim=0)
            v_base = v_base_uncond + cfg_scale * (v_base_cond - v_base_uncond)
            
            # what is your base model computing at this step?
            x_new_in = torch.cat([z + dt * v_base, z + dt * v_base], dim=0)
            t_new_in = t_in + dt
            v_new_base_flat = base_model(x_new_in, t_new_in, y_in, return_hidden=False)
            v_new_base_cond, v_new_base_uncond = v_new_base_flat.chunk(2, dim=0)
            v_new_base = v_new_base_uncond + cfg_scale * (v_new_base_cond - v_new_base_uncond)

        v_draft_flat = draft_model.forward_from_teacher_hidden(v_base_hidden, v_base_emb)
        v_draft_cond, v_draft_uncond = v_draft_flat.chunk(2, dim=0)
        v_draft = v_draft_uncond + cfg_scale * (v_draft_cond - v_draft_uncond)

        loss = F.huber_loss(v_new_base, v_draft, delta=1.0) # model learns to generate the next step from the hidden state here
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
        optimizer.step()

        wandb.log({"loss": loss.item(), "step": train_step})

        if train_step % args.checkpoint_every == 0:
            save_model(draft_model, str(train_step), args.checkpoint_dir)

    save_model(draft_model, "final", args.checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/distill")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    wandb.init(
        project="sit-distill",
        config={
            "SEED": SEED,
            "DEVICE": DEVICE,
            "IMAGE_SIZE": IMAGE_SIZE,
            "LATENT_SIZE": LATENT_SIZE,
            "NUM_CLASSES": NUM_CLASSES,
            "NUM_STEPS": NUM_STEPS,
            "cfg_scale": cfg_scale,
            "num_train_steps": args.num_train_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_every": args.checkpoint_every,
            "lr": args.lr,
        }
    )
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, wandb.run.name)
    main(args)
