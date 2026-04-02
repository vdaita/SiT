from typing import Tuple, List, Dict, Optional
import torch
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import time

np.set_printoptions(suppress=True, precision=9, floatmode='maxprec')

def calculate_residuals(x_new, x_old):
    return torch.mean(torch.abs(x_new - x_old) ** 2, dim=(-3, -2, -1))

def compute_threshold_schedule(timesteps, threshold, dim, num_steps, batch_size):
    sigma_t = (1 - timesteps)
    return (threshold * threshold * (sigma_t ** 2)).unsqueeze(-1).expand(num_steps, batch_size) # we already take them ena

def has_converged(step_residuals, threshold_schedule):
    return (step_residuals < threshold_schedule).all()

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

@dataclass
class SpeculativeTrajectoryResult:
    iters: int
    residual_history: List[List[float]]
    best_draft_indices_history: List[List[int]]

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
    overlap: bool,
    show_progress: bool = False,
    progress_desc: str = "speculative",
):
    device = x.device
    batch_size, channels, height, width = x.shape

    dt = 1.0 / num_steps
    t_traj = torch.arange(0, num_steps, device=device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width, num_steps, batch_size
    )
    # print("Threshold schedule: ", threshold_schedule)
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
        next_draft_traj = draft_traj.clone()
        guaranteed_prefix_len = min(i + 1, num_steps)

        if overlap:
            stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
            with torch.cuda.stream(stream1):
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
                        v_sum = torch.cumsum(v_draft_final[guaranteed_prefix_len - 1:], dim=0)
                        x_draft_traj[guaranteed_prefix_len:] = (
                            x_traj[guaranteed_prefix_len - 1].unsqueeze(0) + v_sum[:-1] * dt
                        )

                    next_draft_traj[j] = x_draft_traj
    
            with torch.cuda.stream(stream2):
                t_base = t_model.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
                y_base = y_traj.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
                y_null_base = y_null_traj.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
                v_base_final = model_call_cfg(
                    base_model,
                    draft_traj,
                    t_base,
                    y_base,
                    y_null_base,
                    cfg_scale,
                )

            stream1.synchronize()
            stream2.synchronize()
            draft_traj = next_draft_traj
        else:
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
                    v_sum = torch.cumsum(v_draft_final[guaranteed_prefix_len - 1:], dim=0)
                    x_draft_traj[guaranteed_prefix_len:] = (
                        x_traj[guaranteed_prefix_len - 1].unsqueeze(0) + v_sum[:-1] * dt
                    )

                next_draft_traj[j] = x_draft_traj
            draft_traj = next_draft_traj
            t_base = t_model.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
            y_base = y_traj.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
            y_null_base = y_null_traj.unsqueeze(0).expand(num_draft_steps + 1, num_steps, batch_size)
            v_base_final = model_call_cfg(
                base_model,
                draft_traj,
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

        residual = torch.mean(torch.square(x_base_traj - draft_traj), dim=(1, -1, -2, -3))
        draft_residual_grid_history.append(residual.detach().cpu())
        best_draft_indices = torch.argmin(residual, dim=0)
        best_draft_indices_history.append(best_draft_indices.detach().cpu().flatten().tolist()) # [batch_size]
        
        step_indices = torch.arange(num_steps, device=device).unsqueeze(1).expand(num_steps, batch_size)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(num_steps, batch_size)
        best_trajectory = x_base_traj[best_draft_indices.unsqueeze(0).expand(num_steps, batch_size), step_indices, batch_indices]
        best_trajectory[:guaranteed_prefix_len] = x_traj[:guaranteed_prefix_len]

        step_residuals = calculate_residuals(best_trajectory, x_traj)
        step_residual = torch.max(step_residuals)
        residual_history.append(step_residuals.detach().cpu().numpy().flatten().tolist())
        x_traj = best_trajectory
        
        draft_traj[0] = x_traj

        if has_converged(step_residuals, threshold_schedule):
            break

    return x_traj[-1], SpeculativeTrajectoryResult(
        iters=outer_iters,
        residual_history=residual_history,
        best_draft_indices_history=best_draft_indices_history
    )

@dataclass
class PicardResult:
    residual_history: List[List[float]]
    threshold_schedule: List[float]
    iters: List[int] # of length num_thresholds
    durations: List[float] # of length num_thresholds

@dataclass
class PiecewisePicardStageResult:
    num_steps: int
    iterations: int
    group_size: int
    threshold: float
    residual_history: List[List[float]]

@dataclass
class UpscalingPiecewisePicardResult:
    total_iterations: int
    stages: List[PiecewisePicardStageResult]

def piecewise_picard_trajectory(
    model,
    x,
    y,
    y_null,
    num_steps: int,
    cfg_scale: float,
    threshold: float,
    group_size: int, # how many can you infer at a given time?
    prev_x_velocities: Optional[torch.Tensor] = None,
):
    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps
    
    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    
    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width, num_steps, batch_size
    )

    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()

    if prev_x_velocities is None:
        prev_x_velocities = torch.zeros((num_steps - 1, batch_size, channels, height, width))

    x_traj[1:] = x_traj[0] + torch.cumsum(prev_x_velocities, dim=0)

    start_index = 0
    num_iterations = 0
    residual_history = []

    while start_index < num_steps - 1:
        x_traj_slice = x_traj[start_index : min(num_steps, start_index + group_size)]
        y_traj_slice = y_traj[start_index : min(num_steps, start_index + group_size)]
        y_null_traj_slice = y_null_traj[start_index : min(num_steps, start_index + group_size)]
        t_model_slice = t_model[start_index : min(num_steps, start_index + group_size)]
        threshold_schedule_slice = threshold_schedule[start_index : min(num_steps, start_index + group_size)]

        v_model = model_call_cfg(model, x_traj_slice, t_model_slice, y_traj_slice, y_null_traj_slice, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[:start_index + 1] = x_traj[:start_index + 1] # up to and including the start index the value remains the same

        window_start = start_index + 1
        window_end = min(num_steps, start_index + group_size + 1)
        window_size = window_end - window_start
        x_traj_new[window_start : window_end] = x_traj[start_index] + torch.cumsum(v_model[:window_size], dim=0) * dt

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        residual_history.append(step_residuals.detach().cpu().numpy().flatten().tolist())
        step_residuals_slice = step_residuals[start_index : min(num_steps, start_index + group_size)]
        mask = step_residuals_slice >= threshold_schedule_slice
        mask_nonzero = torch.nonzero(mask, as_tuple=False)
        idx = mask_nonzero[0][0].item() if mask.any() else -1 # TODO: handle the indexing in a way that supports more images
        
        increment_amount = min(num_steps, start_index + group_size) - start_index
        if idx != -1:
            assert isinstance(idx, int), "idx (from the mask and what not) must be an int"
            increment_amount = max(idx - 1, 0) # the point of this is to keep the last converged as the starting point for the next sequence
        start_index += increment_amount
        if increment_amount > 0:
            x_traj_new[start_index + 1:] = x_traj_new[start_index] + torch.cumsum(prev_x_velocities[start_index:], dim=0) # we know that start_index is correct, what is the index that we have from the previous trajectory starting here?

        x_traj = x_traj_new
        num_iterations += 1

    return x_traj, PiecewisePicardStageResult(
        num_steps=num_steps,
        iterations=num_iterations,
        group_size=group_size,
        threshold=threshold,
        residual_history=residual_history,
    )

def get_interp_velocities(
    trajectory: torch.Tensor,
    multiple: int
):
    num_steps, batch_size, channels, height, width = trajectory.shape
    velocities = trajectory[1:] - trajectory[:-1] # (num_steps - 1) elements
    new_velocities = torch.zeros((num_steps - 1) * multiple, batch_size, channels, height, width, dtype=trajectory.dtype, device=trajectory.device)
    for i in range(num_steps - 1):
        for j in range(multiple):
            new_velocities[multiple * i + j] = velocities[i] / multiple
    return new_velocities

def upscaling_piecewise_picard(
    model,
    x,
    y,
    y_null,
    num_steps_init: int,
    multiples: List[int], # how many multiples do you want to increase this by?
    cfg_scale: float,
    threshold: float,
    group_size: int,
):
    # calculate the first trajectory
    x_traj, initial_stage = piecewise_picard_trajectory(model=model, x=x, y=y, y_null=y_null, num_steps=num_steps_init, cfg_scale=cfg_scale, threshold=threshold, group_size=group_size)
    stages = [initial_stage]
    num_iterations_total = initial_stage.iterations
    curr_num_steps = num_steps_init

    for multiple in multiples:
        x_velocities = get_interp_velocities(x_traj, multiple)
        curr_num_steps = (curr_num_steps - 1) * multiple + 1
        x_traj, stage_result = piecewise_picard_trajectory(model=model, x=x, y=y, y_null=y_null, num_steps=curr_num_steps, cfg_scale=cfg_scale, threshold=threshold, group_size=group_size, prev_x_velocities=x_velocities)
        stages.append(stage_result)
        num_iterations_total += stage_result.iterations
    
    return x_traj[-1], UpscalingPiecewisePicardResult(
        total_iterations=num_iterations_total,
        stages=stages,
    )
    
def picard_trajectory(
    model,
    x,
    y,
    y_null,
    num_steps: int,
    cfg_scale: float,
    thresholds: List[float] | float,
    show_progress: bool = False,
    progress_desc: str = "picard",
):
    t0 = time.perf_counter()

    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)

    if isinstance(thresholds, (float, int)):
        thresholds = [float(thresholds)]

    min_threshold = min(thresholds)
    threshold_schedule = compute_threshold_schedule(
        t_traj, min_threshold, batch_size * channels * height * width, num_steps, batch_size
    )

    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()

    step_residuals = torch.tensor(float("inf"), device=x.device)

    residual_history = []
    step_residual_history = []

    iter_range = range(num_steps)
    if show_progress:
        iter_range = tqdm(iter_range, desc=progress_desc, leave=True)

    cumtime_iters = []
    for i in iter_range:
        v_final = model_call_cfg(model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt

        step_residuals = calculate_residuals(x_traj, x_traj_new)
        step_residual_history.append(step_residuals)
        residual_history.append(step_residuals.cpu().numpy().flatten().flatten().tolist())
        x_traj = x_traj_new

        cumtime_iters.append(time.perf_counter() - t0)
        if has_converged(step_residuals, threshold_schedule):
            break

    num_iters_per_threshold = []
    time_per_threshold = []
    
    threshold_schedule_list = []

    for threshold in thresholds:
        threshold_schedule = compute_threshold_schedule(
            t_traj, threshold, batch_size * channels * height * width, num_steps, batch_size
        )
        threshold_schedule_cpu = threshold_schedule.detach().cpu().numpy().flatten().tolist()

        curr_num_iters = 0
        while curr_num_iters < len(step_residual_history):
            if has_converged(step_residual_history[curr_num_iters], threshold_schedule):
                break
            curr_num_iters += 1

        capped_idx = min(curr_num_iters, len(cumtime_iters) - 1)
        num_iters_per_threshold.append(capped_idx + 1)
        time_per_threshold.append(cumtime_iters[capped_idx])

        threshold_schedule_list.append(threshold_schedule_cpu)

    return x_traj[-1], PicardResult(
        residual_history=residual_history,
        threshold_schedule=threshold_schedule_list,
        iters=num_iters_per_threshold,
        durations=time_per_threshold
    )

@dataclass
class TwoPicardResult:
    base_iters: int
    draft_iters: int
    draft_residual_history: List[List[float]]
    base_residual_history: List[List[float]]

def two_picard_trajectory(
    base_model,
    draft_model,
    x,
    y,
    y_null,
    num_steps: int,
    num_draft_steps: int,
    cfg_scale: float,
    threshold: float,
    draft_threshold: float | None = None,
    show_progress: bool = False,
):
    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    threshold_schedule = compute_threshold_schedule(
        t_traj, threshold, batch_size * channels * height * width, num_steps, batch_size,
    )
    if draft_threshold is None:
        draft_threshold = threshold
    draft_threshold_schedule = compute_threshold_schedule(
        t_traj, draft_threshold, batch_size * channels * height * width, num_steps, batch_size
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

    draft_range = range(min(num_steps, num_draft_steps))
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
        draft_residual_history.append(step_residuals.cpu().numpy().flatten().tolist())
        x_traj = x_traj_new
        if has_converged(step_residuals, draft_threshold_schedule):
            break

    for i in base_range:
        base_iters = i + 1
        v_final = model_call_cfg(base_model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt
        
        step_residuals = calculate_residuals(x_traj, x_traj_new)
        base_residual_history.append(step_residuals.cpu().numpy().flatten().tolist())
        x_traj = x_traj_new
        if has_converged(step_residuals, threshold_schedule):
            break

    return x_traj[-1], TwoPicardResult(
        draft_iters=draft_iters,
        base_iters=base_iters,
        draft_residual_history=draft_residual_history,
        base_residual_history=base_residual_history
    )

def two_picard_trajectory_grid(
    base_model,
    draft_model,
    x,
    y,
    y_null,
    num_steps: int,
    picard_iteration_pairs: List[Tuple[int, int]],
    cfg_scale: float,
    show_progress: bool = False,
):
    batch_size, channels, height, width = x.shape
    dt = 1.0 / num_steps

    t_traj = torch.arange(0, num_steps, device=x.device, dtype=x.dtype) / num_steps
    t_model = t_traj.unsqueeze(-1).expand(num_steps, batch_size)
    y_traj = y.unsqueeze(0).expand(num_steps, batch_size)
    y_null_traj = y_null.unsqueeze(0).expand(num_steps, batch_size)

    x_traj_0 = x.unsqueeze(0).expand(num_steps, batch_size, channels, height, width)
    x_traj = x_traj_0.clone()
    
    max_num_draft_steps = max([e[0] for e in picard_iteration_pairs])
    draft_range = range(max_num_draft_steps)
    base_range = range(num_steps)
    if show_progress:
        draft_range = tqdm(draft_range, desc="draft picard", leave=True)
        base_range = tqdm(base_range, desc="base picard", leave=True)
    
    intermediate_draft_results = {} # {int (number of draft results) -> Tensor (results tensor)}
    possible_draft_steps = list(set([e[0] for e in picard_iteration_pairs]))

    for i in draft_range:
        draft_iters = i + 1
        v_final = model_call_cfg(draft_model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
        x_traj_new = x_traj_0.clone()
        x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt

        if draft_iters in possible_draft_steps:
            intermediate_draft_results[draft_iters] = x_traj_new.clone()
        
        x_traj = x_traj_new

    draft_iter_base_iter_map = {}
    for p in picard_iteration_pairs:
        if not (p[0] in draft_iter_base_iter_map):
            draft_iter_base_iter_map[p[0]] = []
        draft_iter_base_iter_map[p[0]].append(p[1])

    pair_results: Dict[Tuple[int, int], torch.Tensor] = {} 
    
    for draft_iter_count in draft_iter_base_iter_map:
        base_iter_counts = draft_iter_base_iter_map[draft_iter_count]
        max_base_iter_counts = max(base_iter_counts)
        
        x_traj = intermediate_draft_results[draft_iter_count]
        for bi in range(max_base_iter_counts):
            base_iters = bi + 1

            v_final = model_call_cfg(base_model, x_traj, t_model, y_traj, y_null_traj, cfg_scale)
            x_traj_new = x_traj_0.clone()
            x_traj_new[1:] = x_traj_new[1:] + torch.cumsum(v_final[:-1], dim=0) * dt
            x_traj = x_traj_new
            
            pair_results[(draft_iter_count, base_iters)] = x_traj[-1]  

    return pair_results
