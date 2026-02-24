"""
Distill latents from the largest model to the smallest model.
"""
from argparse import Namespace
import argparse
import wandb
from models import SiT_XL_2, SiT_S_2, SiT, SiT_XL_2_short
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

    draft_model = SiT_XL_2_short(input_size=LATENT_SIZE).to(DEVICE)
    draft_model.load_state_dict(find_model("models/XL.pt"), strict=False)
    draft_model.train()
    draft_model.final_layer.requires_grad_(False) # make sure this remains frozen

    # draft_model = SiT_S_2(input_size=LATENT_SIZE).to(DEVICE)
    # draft_model.load_state_dict(find_model("models/S.pt"))
    # draft_model.train()

    # move the unpatchify step from XL to S 

    optimizer = torch.optim.Adam(draft_model.parameters(), lr=args.lr)

    dt = 1.0 / NUM_STEPS

    for train_step in tqdm(range(args.num_train_steps)):
        y = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=DEVICE)
        y_null = torch.full((args.batch_size,), 1000, device=DEVICE)

        x0 = torch.randn(
            args.batch_size, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE
        )

        t_traj = torch.arange(
            0, NUM_STEPS, device=DEVICE, dtype=torch.float32
        ) / NUM_STEPS
        t_traj = t_traj.unsqueeze(-1).expand(NUM_STEPS, args.batch_size)

        y_traj = y.unsqueeze(0).expand(NUM_STEPS, args.batch_size)
        y_null_traj = y_null.unsqueeze(0).expand(NUM_STEPS, args.batch_size)

        x_traj = x0.unsqueeze(0).expand(NUM_STEPS, *x0.shape).clone()
        total_loss = 0.0
        for picard_iter in range(args.num_picard_iters):
            with torch.no_grad():

                x_model = torch.cat([x_traj, x_traj], dim=1)
                t_model = torch.cat([t_traj, t_traj], dim=1)
                y_model = torch.cat([y_traj, y_null_traj], dim=1)

                S, B2, C, H, W = x_model.shape

                v_base_flat, v_base_hidden = base_model(
                    x_model.reshape(S * B2, C, H, W),
                    t_model.reshape(S * B2),
                    y_model.reshape(S * B2),
                    return_hidden=True
                ).reshape(S, B2, C, H, W)

                v_base_uncond = v_base_flat[:, :args.batch_size]
                v_base_cond = v_base_flat[:, args.batch_size:]

                v_base = (
                    v_base_uncond
                    + cfg_scale * (v_base_cond - v_base_uncond)
                )
            optimizer.zero_grad()

            v_draft_flat, v_draft_hidden = draft_model(
                x_model.reshape(S * B2, C, H, W),
                t_model.reshape(S * B2),
                y_model.reshape(S * B2),
                return_hidden=True
            ).reshape(S, B2, C, H, W)

            loss = F.huber_loss(v_base_hidden, v_draft_hidden)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                x_traj_new = x0.unsqueeze(0).expand(NUM_STEPS, *x0.shape).clone()
                x_traj_new[1:] = (
                    x_traj_new[1:]
                    + torch.cumsum(v_base[:-1], dim=0) * dt
                )
                x_traj = x_traj_new


        wandb.log(
            {
                "loss": total_loss / args.num_picard_iters,
                "step": train_step,
            }
        )

        if train_step % args.checkpoint_every == 0:

            save_model(
                draft_model,
                f"{train_step}",
                args.checkpoint_dir,
            )


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