"""
Distill latents from the largest model to the smallest model.
"""
from argparse import Namespace
import argparse
import wandb
from models import SiT_XL_2, SiT_S_2, SiT, SiT_XL_2_short, SiT_B_2, SiT_B_2_short
from download import find_model
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
LATENT_SIZE = IMAGE_SIZE // 8
NUM_CLASSES = 10
NUM_STEPS = 16
cfg_scale = 4.0
hidden_weight_lambda = 0.1

torch.manual_seed(SEED)

def save_model(model: SiT, iter_num: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    filename = f"draft_model_{iter_num}.pt"
    path = os.path.join(folder, filename)
    torch.save(model.state_dict(), path)

@dataclass
class DataInputs:
    z: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor

@dataclass
class DataResult:
    hidden_states: List[torch.Tensor]
    output: torch.Tensor
    input: DataInputs

def generate_picard_iteration_data(model, z, y, num_steps) -> List[DataResult]:
    batch_size, in_channel, latent_size, _ = z.shape

    z_0_traj = z.unsqueeze(0).expand(num_steps, batch_size, in_channel, latent_size, latent_size)
    z_model = z_0_traj.clone()
    y_model = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_model = torch.full((num_steps, batch_size), 1000, device=z.device)
    t_model = torch.arange(0, num_steps, device=z.device).unsqueeze(-1).expand(num_steps, batch_size)

    dt = 1 / NUM_STEPS

    history = []
    with torch.no_grad():
        for step in range(num_steps // 2):
            z_in = torch.cat([z_model, z_model], dim=1)
            t_in = torch.cat([t_model, t_model], dim=1)
            y_in = torch.cat([y_model, y_null_model], dim=1)
            
            z_in_flat, t_in_flat, y_in_flat = z_in.reshape(-1, in_channel, latent_size, latent_size), \
                t_in.reshape(-1), \
                y_in.reshape(-1)

            v_flat, flat_hidden_states = model(z_in_flat, t_in_flat, y_in_flat, return_hidden=True)
            
            v_model = v_flat.reshape(num_steps, batch_size * 2, in_channel, latent_size, latent_size)
            
            v_cond, v_uncond = v_model[:, :batch_size], v_model[:, batch_size:]
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
            z_model = z_0_traj.clone()
            z_model[1:] += torch.cumsum(v[:-1], dim=1) * dt

            history.append(
                DataResult(
                    input=DataInputs(z=z_in_flat, t=t_in_flat, y=y_in_flat),
                    output=v_flat,
                    hidden_states=flat_hidden_states
                )
            )
    return history

def main(args: Namespace):
    base_model = SiT_B_2(input_size=LATENT_SIZE).to(DEVICE)
    base_model.load_state_dict(find_model("models/B.pt"))
    base_model.eval()
    base_model.requires_grad_(False)

    draft_model = SiT_B_2_short(input_size=LATENT_SIZE).to(DEVICE)
    draft_model.train()
    draft_model.requires_grad_(True)

    optimizer = torch.optim.Adam(draft_model.parameters(), lr=args.lr)

    for train_step in tqdm(range(args.num_train_steps)):
        z = torch.randn(args.batch_size, 4, LATENT_SIZE, LATENT_SIZE, device=DEVICE)
        y = torch.randint(0, NUM_CLASSES, (args.batch_size,), device=DEVICE)
        history = generate_picard_iteration_data(base_model, z, y, NUM_STEPS)

        optimizer.zero_grad()
        
        total_hidden_loss = 0
        total_vel_loss = 0
        total_loss = 0

        for data_slice in history:
            draft_model_v, draft_model_hidden_states = draft_model(data_slice.input.z, data_slice.input.t, data_slice.input.y, return_hidden=True)
            
            vel_loss = F.mse_loss(draft_model_v, data_slice.output)
            
            hidden_loss = 0
            for draft_h, base_h in zip(draft_model_hidden_states, data_slice.hidden_states[::4]):
                hidden_loss += hidden_weight_lambda * F.mse_loss(draft_h, base_h)
                assert draft_h.shape == base_h.shape
            
            loss = vel_loss + hidden_loss
            total_vel_loss += vel_loss.item()
            total_hidden_loss += hidden_loss.item() # type: ignore
            total_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
        optimizer.step()

        norm = (NUM_STEPS // 2) * args.batch_size
        wandb.log({ # type: ignore
            "loss": total_loss / norm,
            "vel_loss": total_vel_loss / norm,
            "hidden_loss": total_hidden_loss / norm,
            "step": train_step,
        })

        if train_step % args.checkpoint_every == 0:
            save_model(draft_model, str(train_step), args.checkpoint_dir)
    save_model(draft_model, "final", args.checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
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