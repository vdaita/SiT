import torch
from tqdm import tqdm
import numpy as np

np.set_printoptions(suppress=True, precision=9, floatmode='maxprec')

def calculate_residuals(x_new, x_old):
    return torch.mean(torch.abs(x_new - x_old) ** 2, dim=(-3, -2, -1))

def compute_threshold_schedule(timesteps, threshold, dim):
    return torch.tensor(threshold) 

def has_converged(step_residuals, threshold_schedule):
    threshold_grid = torch.tensor(threshold_schedule).unsqueeze(-1).expand_as(step_residuals).to(step_residuals.device)
    return (step_residuals < threshold_grid).all()

def model_call_cfg(model, x, t, y, y_null, cfg_scale: float):
    batch_size = x.shape[-4]
    channels, height, width = x.shape[-3:]
    leading_shape = x.shape[:-4]

    x_model = torch.cat([x, x], dim=-4)
    t_model = torch.cat([t, t], dim=-1)
    y_model = torch.cat([y, y_null], dim=-1)

    x_model_flat = x_model.reshape(-1, channels, height, width)
    t_model_flat = t_model.reshape(-1)
    y_model_flat = y_model.reshape(-1)

    first_param = next(model.parameters(), None)
    if first_param is not None:
        model_device = first_param.device
        model_dtype = first_param.dtype
        x_model_flat = x_model_flat.to(device=model_device, dtype=model_dtype)
        t_model_flat = t_model_flat.to(device=model_device, dtype=model_dtype)
        y_model_flat = y_model_flat.to(device=model_device)

    v_flat = model(x_model_flat, t_model_flat, y_model_flat)
    v_flat = v_flat.to(device=x.device, dtype=x.dtype)
    v = v_flat.reshape(*leading_shape, 2, batch_size, channels, height, width)
    v_cond = v.select(dim=-5, index=0)
    v_uncond = v.select(dim=-5, index=1)
    return v_uncond + cfg_scale * (v_cond - v_uncond)

def speculative_trajectory(
    base_model,
    draft_model,
    x,
    y,
    y_null,
    num_steps: int,
    num_draft_steps: int,
    cfg_scale: float,
    threshold: float,
    show_progress: bool = False,
    progress_desc: str = "speculative",
):
    device = x.device
    batch_size, channels, height, width = x.shape

    dt = 1.0 / num_steps
    t_traj = torch.arange(0, num_steps, device=device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width
    )
    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()

    draft_traj = torch.zeros(
        (num_draft_steps + 1, num_steps, batch_size, channels, height, width),
        device=device,
        dtype=x.dtype,
    )
    draft_traj[0] = x_traj_0

    outer_iters = 0
    step_residual = torch.tensor(float("inf"), device=device)
    residual_history = []
    draft_residual_grid_history = []
    best_draft_indices_history = []

    iter_range = range(num_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc=progress_desc, leave=False)

    for i in iter_range:
        outer_iters = i + 1
        prev_draft_traj = draft_traj.clone()
        next_draft_traj = prev_draft_traj.clone()
        guaranteed_prefix_len = min(i + 1, num_steps)

        for j in range(1, num_draft_steps + 1):
            v_draft_final = model_call_cfg(
                draft_model,
                next_draft_traj[j - 1],
                t_model,
                y_traj,
                y_null_traj,
                cfg_scale,
            )

            x_draft_traj = x_traj_0.clone()
            x_draft_traj[:guaranteed_prefix_len] = x_traj[:guaranteed_prefix_len]
            if guaranteed_prefix_len < num_steps:
                v_draft_suffix_sum = torch.cumsum(v_draft_final[guaranteed_prefix_len - 1 :], dim=0)
                x_draft_traj[guaranteed_prefix_len:] = (
                    x_traj[guaranteed_prefix_len - 1].unsqueeze(0)
                    + v_draft_suffix_sum[:-1] * dt
                )

            next_draft_traj[j] = x_draft_traj
        t_base = t_model.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
        y_base = y_traj.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
        y_null_base = y_null_traj.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
        v_base_final = model_call_cfg(
            base_model,
            next_draft_traj,
            t_base,
            y_base,
            y_null_base,
            cfg_scale,
        )

        x_base_traj = x_traj_0.unsqueeze(0).expand(num_draft_steps + 1, num_steps, *x.shape).clone()
        x_base_traj[:, :guaranteed_prefix_len] = x_traj[:guaranteed_prefix_len].unsqueeze(0)
        if guaranteed_prefix_len < num_steps:
            v_base_suffix_sum = torch.cumsum(v_base_final[:, guaranteed_prefix_len - 1 :], dim=1)
            x_base_traj[:, guaranteed_prefix_len:] = (
                x_traj[guaranteed_prefix_len - 1].unsqueeze(0).unsqueeze(0)
                + v_base_suffix_sum[:, :-1] * dt
            )

        residual = torch.mean(torch.abs(x_base_traj - next_draft_traj), dim=(-1, -2, -3))
        draft_residual_grid_history.append(residual.detach().cpu().flatten())
        best_draft_indices = torch.argmin(residual, dim=0)
        best_draft_indices_history.append(best_draft_indices.detach().cpu().flatten())
        step_indices = torch.arange(num_steps, device=device).unsqueeze(1).expand(num_steps, batch_size)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(num_steps, batch_size)
        best_trajectory = x_base_traj[best_draft_indices, step_indices, batch_indices]
        best_trajectory[:guaranteed_prefix_len] = x_traj[:guaranteed_prefix_len]

        step_residuals = calculate_residuals(best_trajectory, x_traj)
        step_residual = torch.max(step_residuals)
        residual_history.append(step_residuals.detach().cpu().numpy().flatten())
        x_traj = best_trajectory
        next_draft_traj[0] = x_traj
        draft_traj = next_draft_traj

        if has_converged(step_residuals, threshold_schedule):
            break

    return x_traj[-1], {
        "iters": outer_iters,
        "residual": float(step_residual.detach().cpu().item()),
        "residual_history": residual_history,
        "best_draft_indices_history": best_draft_indices_history,
        "draft_residual_grid_history": draft_residual_grid_history,
        "thresholds": threshold_schedule.detach().cpu().numpy().flatten().tolist(),
    }


def picard_trajectory(
    model,
    x,
    y,
    y_null,
    num_steps: int,
    cfg_scale: float,
    threshold: float,
    show_progress: bool = False,
    progress_desc: str = "picard",
):
    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width
    )
    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()
    step_residuals = torch.tensor(float("inf"), device=x.device)
    iters = 0
    residual_history = []

    iter_range = range(num_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc=progress_desc, leave=True)

    for i in iter_range:
        iters = i + 1
        v_final = model_call_cfg(model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        residual_history.append(step_residuals.cpu().numpy().flatten())
        x_traj = x_traj_new
        if has_converged(step_residuals, threshold_schedule):
            break

    return x_traj[-1], {
        "iters": iters,
        "residual": float(torch.max(step_residuals).detach().cpu().item()),
        "residual_history": residual_history,
        "thresholds": threshold_schedule.detach().cpu().numpy().flatten().tolist(),
    }


def multi_stage_trajectory(
    model,
    x,
    y,
    y_null,
    num_coarse_steps: int,
    num_fine_steps: int,
    cfg_scale: float,
    threshold: float,
    show_progress: bool = False
):
    batch_size, channels, height, width = x.shape
    
    t_coarse_traj = torch.arange(0, num_coarse_steps, device=x.device, dtype=x.dtype) / num_coarse_steps
    t_coarse_model = t_coarse_traj.unsqueeze(-1).expand(num_coarse_steps, batch_size)

    y_model = y.unsqueeze(0).expand(num_coarse_steps, batch_size)
    y_null_model = y_null.unsqueeze(0).expand(num_coarse_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_coarse_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()

    num_iters = 0
    residual_history = []

    iter_range = range(num_coarse_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc="Coarse steps", leave=True)

    # TODO: run initial forward passes
    for iter_id in iter_range:
        num_iters = iter_id + 1
        # with the current iteration, what should the values be?
        x_traj_updates = x_traj.clone()
        for fine_iter_id in range(num_fine_steps):
            t_fine_model = (fine_iter_id / (num_fine_steps * num_coarse_steps)) + t_coarse_model
            v_fine = model_call_cfg(model, x_traj_updates, t_fine_model, y_model, y_null_model, cfg_scale)
            x_traj_updates += v_fine * (1/(num_fine_steps * num_coarse_steps))
        v_coarse = x_traj_updates - x_traj

        x_traj_next = x_traj_0.clone()
        x_traj_next[1:] = x_traj_0[0].unsqueeze(0) + torch.cumsum(v_coarse[:-1], dim=0)
        
        # calculate the residual and decide whether to stop
        step_residuals = calculate_residuals(x_traj_next, x_traj)
        residual_history.append(step_residuals.cpu().numpy().flatten())
        x_traj = x_traj_next
        if has_converged(step_residuals, threshold):
            break
    
    return x_traj[-1], {
        "iters": num_iters,
        "residual_history": residual_history
    }


def parareal(
    model,
    x,
    y,
    y_null,
    num_steps: int,
    num_fine_steps: int,  # sub-steps per coarse interval for the fine solver
    cfg_scale: float,
    threshold: float,
    show_progress: bool = False,
):
    """
    Parareal algorithm for parallel-in-time ODE solving.
    
    Coarse solver G: one Euler step per coarse timestep (same model, fewer steps).
    Fine solver F: num_fine_steps Euler sub-steps per coarse interval (same model).
    
    Update rule (standard parareal):
        u_{n+1}^{k+1} = F(u_n^{k+1}) + G(u_{n+1}^k) - G(u_n^k)
    """
    batch_size, channels, height, width = x.shape
    dt_coarse = 1.0 / num_steps
    dt_fine = dt_coarse / num_fine_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)

    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width
    )

    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)

    # --- Helper: coarse solver G ---
    # One Euler step with the model at the coarse timestep
    def coarse_solve(x_in):
        # x_in: (num_steps, batch, C, H, W) — state at each coarse node
        v = model_call_cfg(model, x_in, t_model, y_traj, y_null_traj, cfg_scale)
        return x_in + v * dt_coarse  # shape: (num_steps, batch, C, H, W)

    # --- Helper: fine solver F ---
    # num_fine_steps Euler sub-steps within each coarse interval
    def fine_solve(x_in):
        # x_in: (num_steps, batch, C, H, W) — state at each coarse node
        x_fine = x_in.clone()
        for j in range(num_fine_steps):
            t_fine = t_traj + j * dt_fine  # (num_steps,)
            t_fine_model = t_fine.unsqueeze(-1).expand(num_steps, batch_size)
            v = model_call_cfg(model, x_fine, t_fine_model, y_traj, y_null_traj, cfg_scale)
            x_fine = x_fine + v * dt_fine
        return x_fine  # shape: (num_steps, batch, C, H, W)

    # --- Initialise with coarse solver ---
    # Build initial trajectory by sequential coarse propagation from x
    x_traj = x_traj_0.clone()
    x_traj[0] = x  # u_0 is always the initial condition
    g_old = coarse_solve(x_traj)  # G(u_n^0) for all n
    for n in range(1, num_steps):
        x_traj[n] = g_old[n - 1]

    num_iters = 0
    residual_history = []

    iter_range = range(num_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc="parareal", leave=True)

    for k in iter_range:
        num_iters = k + 1

        # Fine solves are embarrassingly parallel across all coarse intervals
        f_new = fine_solve(x_traj)           # F(u_n^k)  for all n
        g_old = coarse_solve(x_traj)         # G(u_n^k)  for all n

        # Build updated trajectory sequentially (parareal correction sweep)
        x_traj_new = x_traj_0.clone()
        x_traj_new[0] = x  # u_0 fixed
        for n in range(1, num_steps):
            # G(u_{n}^{k+1}) computed from newly updated x_traj_new[n-1]
            t_n = t_traj[n - 1].unsqueeze(-1).expand(batch_size)
            t_single = t_n.unsqueeze(0)  # (1, batch)
            x_prev = x_traj_new[n - 1].unsqueeze(0)  # (1, batch, C, H, W)
            y_single = y_traj[n - 1].unsqueeze(0)
            y_null_single = y_null_traj[n - 1].unsqueeze(0)
            v_g_new = model_call_cfg(model, x_prev, t_single, y_single, y_null_single, cfg_scale)
            g_new_n = x_traj_new[n - 1] + v_g_new[0] * dt_coarse

            # Parareal update: F(u_{n-1}^k) + G(u_n^k+1) - G(u_{n-1}^k)
            x_traj_new[n] = f_new[n - 1] + g_new_n - g_old[n - 1]

        step_residuals = calculate_residuals(x_traj_new, x_traj)
        residual_history.append(step_residuals.detach().cpu().numpy().flatten())
        x_traj = x_traj_new

        if has_converged(step_residuals, threshold_schedule):
            break

    return x_traj[-1], {
        "iters": num_iters,
        "residual": float(torch.max(step_residuals).detach().cpu().item()),
        "residual_history": residual_history,
        "thresholds": threshold_schedule.detach().cpu().numpy().flatten().tolist(),
    }
def two_picard_trajectory(
    base_model,
    draft_model,
    x,
    y,
    y_null,
    num_steps: int,
    cfg_scale: float,
    threshold: float,
    show_progress: bool = False,
):
    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width
    )
    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()
    step_residuals = torch.tensor(float("inf"), device=x.device)
    
    draft_iters = 0
    base_iters = 0
    
    draft_residual_history = []
    base_residual_history = []

    draft_range = range(num_steps)
    base_range = range(num_steps)
    if show_progress:
        draft_range = tqdm(draft_range, desc="Draft Picard", leave=True)
        base_range = tqdm(base_range, desc="Base Picard", leave=True)

    for i in draft_range:
        draft_iters = i + 1
        v_final = model_call_cfg(draft_model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        draft_residual_history.append(step_residuals.cpu().numpy().flatten())
        x_traj = x_traj_new
        if has_converged(step_residuals, threshold_schedule):
            break

    for i in base_range:
        base_iters = i + 1
        v_final = model_call_cfg(base_model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt
        
        step_residuals = calculate_residuals(x_traj, x_traj_new)
        base_residual_history.append(step_residuals.cpu().numpy().flatten())
        x_traj = x_traj_new
        if has_converged(step_residuals, threshold_schedule):
            break

    return x_traj[-1], {
        "base_iters": base_iters,
        "draft_iters": draft_iters,
        "residual": float(torch.max(step_residuals).detach().cpu().item()),
        "draft_residual_history": draft_residual_history,
        "base_residual_history": base_residual_history,
        "thresholds": threshold_schedule.detach().cpu().numpy().flatten().tolist(),
    }

def straight_line_speculation(
    model,
    x,
    y,
    y_null,
    num_steps: int,
    num_trajs: int,
    cfg_scale: float,
    threshold: float,
    show_progress: bool = False,
    progress_desc: str = "straight_line_picard",
):
    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).unsqueeze(0).expand(num_trajs, num_steps, batch_size)
    y_traj = y.unsqueeze(0).unsqueeze(0).expand(num_trajs, num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).unsqueeze(0).expand(num_trajs, num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).unsqueeze(0).expand(num_trajs, num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()
    iters = 0
    residual_history = []
    best_indices_history = []

    iter_range = range(num_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc=progress_desc, leave=True)

    for i in iter_range:
        iters = i + 1
        v_pred = model_call_cfg(model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[:, 1:] = x_traj_new[:, 1:] + torch.cumsum(v_pred[:, :-1], dim=1) * dt

        # figure out the Picard defect to determine the trajectory
        residuals = calculate_residuals(x_traj, x_traj_new)
        residual_history.append(residuals.detach().cpu().flatten(start_dim=-2))
        # print("Residuals: ", residual_history[-1])

        # figure out if the trajectories are satisfactory
        done = False
        for traj in range(num_trajs):
            if has_converged(residuals[traj], threshold):
                done = True
                break
        if done:
            break

        # trajectories are bad, select the best one
        best_traj = torch.argmin(torch.mean(residuals, dim=1)) # averaging across the number of steps, select the trajectory with the lowest average residual
        best_indices_history.append(best_traj.detach().cpu())

        x_traj_proposals = torch.zeros_like(x_traj_0)
        x_traj_proposals[0] = x_traj_new[best_traj]
        v_best = v_pred[best_traj]
        for traj in range(1, num_trajs):
            # predict a new "straight line" period as a proposal
            line_start = i
            line_end = line_start + ((num_steps - line_start) // traj) # exclusive, we can't include num_steps itself for example
                        
            x_traj_proposals[traj, :(line_start + 1)] = x_traj_new[best_traj, :(line_start + 1)]

            v_line = v_best[line_start]

            x0 = x_traj_proposals[traj, line_start]

            line = torch.arange(1, line_end - line_start, device=x.device)

            line_proposal = (
                x0.unsqueeze(0)
                + v_line.unsqueeze(0) * line.view(-1,1,1,1,1) * dt
            )

            x_traj_proposals[traj, line_start+1:line_end] = line_proposal


            # tail
            x_traj_proposals[traj, line_end:] = (
                x_traj_proposals[traj, line_end - 1].unsqueeze(0)
                + torch.cumsum(v_best[line_end - 1:-1], dim=0) * dt
            )
        x_traj = x_traj_proposals

    return x_traj[0, -1], {
        "iters": iters,
        "best_indices_history": best_indices_history,
        "residual_history": residual_history,
    }

def sequential_trajectory(
    model,
    x,
    y,
    y_null,
    num_steps: int,
    cfg_scale: float,
    show_progress: bool = False,
    progress_desc: str = "sequential",
):
    dt = 1.0 / num_steps
    x_seq = x.clone()

    iter_range = range(num_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc=progress_desc, leave=False)

    for i in iter_range:
        t_val = i / num_steps
        t_th = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
        v_final = model_call_cfg(model, x_seq, t_th, y, y_null, cfg_scale)
        x_seq = x_seq + v_final * dt

    return x_seq, {"iters": num_steps, "residual": 0.0, "residual_history": []}
