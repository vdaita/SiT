"""
plot_results.py
===============
Reads from:
  outputs/results_baseline.json
  outputs/results_two_picard.json
  outputs/results_speculative.json

Produces in outputs/plots/:
  01_baseline_iters.png
  01_baseline_wallclock.png
  02_twopic_iters_tau{tau}.png
  02_twopic_wallclock_tau{tau}.png
  03_spec_acceptance_tau{tau}_K{K}.png
  04_spec_wallclock_tau{tau}_K{K}.png
  05_residual_heatmap_{pair}_{overlap_str}.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILES = {
    "baseline":    "outputs/results_baseline.json",
    "two_picard":  "outputs/results_two_picard.json",
    "speculative": "outputs/results_speculative.json",
}
PLOTS_DIR   = "outputs/plots"
MODEL_ORDER = ["S", "B", "L", "XL"]

os.makedirs(PLOTS_DIR, exist_ok=True)

def load(kind):
    p = RESULTS_FILES[kind]
    if not os.path.exists(p):
        print(f"  [warn] {p} not found, skipping.")
        return {}
    with open(p) as f:
        return json.load(f)

def ms(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if len(a) == 0:
        return np.nan, 0.0
    return float(np.mean(a)), float(np.std(a))

def sorted_unique(keys, extract, cast=str):
    return sorted(set(cast(extract(k)) for k in keys))

baseline    = load("baseline")
two_picard  = load("two_picard")
speculative = load("speculative")

# Discover axes from actual keys present in each file
def baseline_models():
    return sorted(set(k.split("_")[0] for k in baseline),
                  key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

def baseline_steps():
    return sorted(set(int(k.split("_steps")[1].split("_")[0]) for k in baseline))

def baseline_taus():
    return sorted(set(float(k.split("_tau")[1]) for k in baseline))

def twopic_pairs():
    return sorted(set(k.split("_steps")[0] for k in two_picard))

def twopic_steps():
    return sorted(set(int(k.split("_steps")[1].split("_")[0]) for k in two_picard))

def twopic_taus():
    return sorted(set(float(k.split("_tau")[1].split("_")[0]) for k in two_picard))

def twopic_dinits():
    return sorted(set(int(k.split("_dinit")[1]) for k in two_picard))

def spec_pairs():
    # key format: {draft}to{base}_steps{N}_tau{tau}_K{K}_{overlap_str}
    return sorted(set(k.split("_steps")[0] for k in speculative))

def spec_steps():
    return sorted(set(int(k.split("_steps")[1].split("_")[0]) for k in speculative))

def spec_taus():
    return sorted(set(float(k.split("_tau")[1].split("_")[0]) for k in speculative))

def spec_Ks():
    return sorted(set(int(k.split("_K")[1].split("_")[0]) for k in speculative))

# ------------------------------------------------------------------ #
# Fig 1: Baseline iters + wall-clock
# ------------------------------------------------------------------ #
if baseline:
    models   = baseline_models()
    steps_l  = baseline_steps()
    taus_l   = baseline_taus()

    for metric, label, color, fname in [
        ("iters",        "Picard iters",   "steelblue",  "01_baseline_iters.png"),
        ("wall_clock_s", "Wall-clock (s)", "darkorange", "01_baseline_wallclock.png"),
    ]:
        fig, axes = plt.subplots(
            len(steps_l), len(taus_l),
            figsize=(4 * len(taus_l), 3.5 * len(steps_l)),
            squeeze=False,
        )
        fig.suptitle(f"Baseline: {label}", fontsize=13, fontweight="bold")

        for row, num_steps in enumerate(steps_l):
            for col, tau in enumerate(taus_l):
                ax = axes[row][col]
                means, stds = [], []
                for m in models:
                    key = f"{m}_steps{num_steps}_tau{tau}"
                    if key not in baseline:
                        means.append(np.nan); stds.append(0.0); continue
                    mv, sv = ms([r[metric] for r in baseline[key]])
                    means.append(mv); stds.append(sv)

                xs = np.arange(len(models))
                ax.bar(xs, means, yerr=stds, color=color, capsize=4)
                ax.set_xticks(xs); ax.set_xticklabels(models, fontsize=8)
                ax.set_title(f"steps={num_steps}  τ={tau}", fontsize=8)
                if col == 0:
                    ax.set_ylabel(label, fontsize=8)

        plt.tight_layout()
        path = f"{PLOTS_DIR}/{fname}"
        plt.savefig(path, dpi=150); plt.close()
        print(f"Saved {path}")

# ------------------------------------------------------------------ #
# Fig 2a: Two-picard stacked iters, one figure per tau
# ------------------------------------------------------------------ #
if two_picard:
    pairs_l  = twopic_pairs()
    steps_l  = twopic_steps()
    taus_l   = twopic_taus()
    dinits_l = twopic_dinits()

    for tau in taus_l:
        n_pairs = len(pairs_l)
        n_steps = len(steps_l)
        if not n_pairs:
            continue
        fig, axes = plt.subplots(n_steps, n_pairs,
                                 figsize=(5 * n_pairs, 4 * n_steps),
                                 squeeze=False)
        fig.suptitle(f"Two-picard: iters breakdown  tau={tau}",
                     fontsize=12, fontweight="bold")

        for row, num_steps in enumerate(steps_l):
            for col, pair in enumerate(pairs_l):
                ax = axes[row][col]
                base_name = pair.split("to")[1]

                # --- column 0: baseline solo picard iters ---
                bkey       = f"{base_name}_steps{num_steps}_tau{tau}"
                base_total = int(round(np.mean([r["iters"] for r in baseline[bkey]])))                              if bkey in baseline else 0

                # --- columns 1+: two-picard at each valid dinit ---
                valid_dinits = [n for n in dinits_l if n <= num_steps]
                all_labels   = ["baseline"] + [str(n) for n in valid_dinits]

                # Baseline bar: no draft, all base
                all_d = [0]
                all_b = [base_total]

                for n in valid_dinits:
                    key = f"{pair}_steps{num_steps}_tau{tau}_dinit{n}"
                    if key not in two_picard:
                        all_d.append(0); all_b.append(0); continue
                    all_d.append(int(round(np.mean([r["draft_iters"] for r in two_picard[key]]))))
                    all_b.append(int(round(np.mean([r["base_iters"]  for r in two_picard[key]]))))

                xs = np.arange(len(all_labels))
                ax.bar(xs, all_d, label="Draft", color="steelblue")
                ax.bar(xs, all_b, bottom=all_d, label="Base", color="darkorange")
                ax.set_xticks(xs)
                ax.set_xticklabels(all_labels, fontsize=8)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.set_xlabel("Max draft iters")
                if col == 0:
                    ax.set_ylabel(f"steps={num_steps}\nPicard iters", fontsize=9)
                ax.set_title(pair.replace("to", " -> "), fontsize=9)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)

        plt.tight_layout()
        path = f"{PLOTS_DIR}/02_twopic_iters_tau{tau}.png"
        plt.savefig(path, dpi=150); plt.close()
        print(f"Saved {path}")

# ------------------------------------------------------------------ #
# Fig 2b: Two-picard wall-clock ratio vs baseline — per-example boxplot
# y-axis: wall_clock(two_picard) / wall_clock(baseline) as % per image
# x-axis: only dinit values <= num_steps for that row
# ------------------------------------------------------------------ #
if two_picard and baseline:
    for tau in twopic_taus():
        pairs_l  = twopic_pairs()
        steps_l  = twopic_steps()
        dinits_l = twopic_dinits()
        if not pairs_l:
            continue

        fig, axes = plt.subplots(len(steps_l), len(pairs_l),
                                 figsize=(5 * len(pairs_l), 4 * len(steps_l)),
                                 squeeze=False)
        fig.suptitle(f"Two-picard wall-clock % of baseline  tau={tau}",
                     fontsize=12, fontweight="bold")

        for row, num_steps in enumerate(steps_l):
            for col, pair in enumerate(pairs_l):
                ax = axes[row][col]
                base_name = pair.split("to")[1]
                bkey      = f"{base_name}_steps{num_steps}_tau{tau}"

                if bkey not in baseline:
                    ax.set_visible(False); continue

                base_by_idx = {r["img_idx"]: r["wall_clock_s"]
                               for r in baseline[bkey]}

                # only dinit values that are <= num_steps
                valid_dinits = [n for n in dinits_l if n <= num_steps]
                if not valid_dinits:
                    ax.set_visible(False); continue

                box_data, labels = [], []
                for n in valid_dinits:
                    key = f"{pair}_steps{num_steps}_tau{tau}_dinit{n}"
                    if key not in two_picard:
                        continue
                    ratios = []
                    for r in two_picard[key]:
                        base_w = base_by_idx.get(r["img_idx"])
                        if base_w and base_w > 0:
                            ratios.append(100.0 * r["wall_clock_s"] / base_w)
                    if ratios:
                        box_data.append(ratios)
                        labels.append(str(n))

                if not box_data:
                    ax.set_visible(False); continue

                bp = ax.boxplot(box_data, patch_artist=True,
                                medianprops=dict(color="black", linewidth=1.5))
                for patch in bp["boxes"]:
                    patch.set_facecolor("steelblue")
                    patch.set_alpha(0.6)

                ax.axhline(100, color="red", linestyle="--", linewidth=1,
                           label="Baseline (100%)")
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_xlabel("Max draft iters")
                if col == 0:
                    ax.set_ylabel(f"steps={num_steps}\n% of baseline wall-clock",
                                  fontsize=8)
                ax.set_title(pair.replace("to", " -> "), fontsize=9)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)

        plt.tight_layout()
        path = f"{PLOTS_DIR}/02_twopic_wallclock_tau{tau}.png"
        plt.savefig(path, dpi=150); plt.close()
        print(f"Saved {path}")

# ------------------------------------------------------------------ #
# Fig 3: Speculative acceptance rate — overlap vs sequential
# ------------------------------------------------------------------ #
if speculative:
    for tau in spec_taus():
        for K in spec_Ks():
            pairs_l = spec_pairs()
            steps_l = spec_steps()
            if not pairs_l:
                continue

            fig, axes = plt.subplots(2, len(pairs_l),
                                     figsize=(5 * len(pairs_l), 7),
                                     squeeze=False)
            fig.suptitle(f"Acceptance rate  τ={tau}  K={K}",
                         fontsize=12, fontweight="bold")

            for col, pair in enumerate(pairs_l):
                for row, (overlap_str, color) in enumerate([
                    ("overlap",    "steelblue"),
                    ("sequential", "darkorange"),
                ]):
                    ax = axes[row][col]
                    plotted_any = False

                    for num_steps, ls in zip(steps_l, ["-", "--", ":", "-."]):
                        key = f"{pair}_steps{num_steps}_tau{tau}_K{K}_{overlap_str}"
                        if key not in speculative:
                            continue
                        all_acc = [r["acceptance_history"] for r in speculative[key]]
                        max_len = max(len(a) for a in all_acc)
                        padded  = np.full((len(all_acc), max_len), np.nan)
                        for i, a in enumerate(all_acc):
                            arr = np.array(a, dtype=float)
                            padded[i, :len(arr)] = (arr > 0).astype(float)
                        mean_acc = np.nanmean(padded, axis=0)
                        std_acc  = np.nanstd(padded, axis=0)
                        iters    = np.arange(1, max_len + 1)
                        overall  = float(np.nanmean(mean_acc))
                        ax.plot(iters, mean_acc, color=color, linestyle=ls,
                                linewidth=1.5,
                                label=f"steps={num_steps} (μ={overall:.2f})")
                        ax.fill_between(iters,
                                        np.clip(mean_acc - std_acc, 0, 1),
                                        np.clip(mean_acc + std_acc, 0, 1),
                                        alpha=0.1, color=color)
                        plotted_any = True

                    ax.set_ylim(0, 1.05)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    ax.set_xlabel("Picard iteration")
                    if col == 0:
                        ax.set_ylabel("Acceptance rate")
                    ax.set_title(f"{pair.replace('to', ' → ')}  [{overlap_str}]",
                                 fontsize=9)
                    if plotted_any:
                        ax.legend(fontsize=6)

            plt.tight_layout()
            path = f"{PLOTS_DIR}/03_spec_acceptance_tau{tau}_K{K}.png"
            plt.savefig(path, dpi=150); plt.close()
            print(f"Saved {path}")

# ------------------------------------------------------------------ #
# Fig 4: Speculative wall-clock: baseline vs overlap vs sequential
# ------------------------------------------------------------------ #
if speculative and baseline:
    for tau in spec_taus():
        for K in spec_Ks():
            pairs_l = spec_pairs()
            steps_l = spec_steps()
            if not pairs_l:
                continue

            fig, axes = plt.subplots(len(steps_l), len(pairs_l),
                                     figsize=(5 * len(pairs_l), 4 * len(steps_l)),
                                     squeeze=False)
            fig.suptitle(f"Speculative wall-clock  τ={tau}  K={K}",
                         fontsize=12, fontweight="bold")

            for row, num_steps in enumerate(steps_l):
                for col, pair in enumerate(pairs_l):
                    ax = axes[row][col]
                    base_name = pair.split("to")[1]

                    candidates = [
                        (f"{base_name}_steps{num_steps}_tau{tau}",
                         "Baseline", "gray", baseline),
                        (f"{pair}_steps{num_steps}_tau{tau}_K{K}_overlap",
                         "Overlap", "steelblue", speculative),
                        (f"{pair}_steps{num_steps}_tau{tau}_K{K}_sequential",
                         "Sequential", "darkorange", speculative),
                    ]

                    bars, labels, colors = [], [], []
                    for key, lbl, clr, src in candidates:
                        if key in src:
                            m, s = ms([r["wall_clock_s"] for r in src[key]])
                            bars.append((m, s))
                            labels.append(lbl)
                            colors.append(clr)

                    if not bars:
                        ax.set_visible(False); continue

                    xs = np.arange(len(bars))
                    ax.bar(xs, [b[0] for b in bars],
                           yerr=[b[1] for b in bars],
                           color=colors, capsize=5)
                    ax.set_xticks(xs)
                    ax.set_xticklabels(labels, fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f"steps={num_steps}\nWall-clock (s)", fontsize=8)
                    ax.set_title(pair.replace("to", " → "), fontsize=9)

            plt.tight_layout()
            path = f"{PLOTS_DIR}/04_spec_wallclock_tau{tau}_K{K}.png"
            plt.savefig(path, dpi=150); plt.close()
            print(f"Saved {path}")

# ------------------------------------------------------------------ #
# Fig 4b: Sequential-evaluation-depth comparison
#
# "Sequential depth" = total base-model-equivalent compute that must
# complete serially before you have the final image.
#
# Baseline Picard:
#   depth = N_iters  (each is one base forward pass)
#
# Sequential speculative:
#   each outer iteration runs K draft steps then 1 base step in series
#   draft cost in base-equiv units = K * (draft_wall / base_wall)
#   depth = sum over iters of (K_draft_used * ratio + 1)
#   — but we only know total iters, not per-iter draft usage, so we
#   approximate: accepted draft steps ≈ acceptance_rate * K * N_iters
#   depth_seq = N_iters + accepted_drafts * ratio
#
# Overlap speculative:
#   base and draft run in parallel each round.
#   serial cost per round = max(1, K * ratio)  [base-equiv units]
#   since draft << base, K*ratio < 1 for most pairs → cost ≈ 1 per round
#   depth_overlap = N_iters * max(1.0, K * ratio)
#
# Wall-clock ratio estimated from baseline results:
#   ratio = mean_wall_per_iter(draft) / mean_wall_per_iter(base)
#   using matching num_steps and tau where available, else nearest.
# ------------------------------------------------------------------ #
if speculative and baseline:

    def base_wall_per_iter(model_name, num_steps, tau):
        """Mean wall-clock per Picard iteration for a single model run."""
        key = f"{model_name}_steps{num_steps}_tau{tau}"
        if key not in baseline:
            return None
        records = baseline[key]
        walls = [r["wall_clock_s"] for r in records]
        iters = [r["iters"] for r in records]
        if not walls or not iters or np.mean(iters) == 0:
            return None
        return float(np.mean(walls) / np.mean(iters))

    def find_ratio(draft_name, base_name, num_steps, tau):
        """draft forward-pass time / base forward-pass time."""
        d = base_wall_per_iter(draft_name, num_steps, tau)
        b = base_wall_per_iter(base_name,  num_steps, tau)
        if d is None or b is None or b == 0:
            return None
        return d / b

    for tau in spec_taus():
        for K in spec_Ks():
            pairs_l = spec_pairs()
            steps_l = spec_steps()
            if not pairs_l:
                continue

            fig, axes = plt.subplots(len(steps_l), len(pairs_l),
                                     figsize=(5 * len(pairs_l), 4 * len(steps_l)),
                                     squeeze=False)
            fig.suptitle(
                f"Depth ratio vs baseline Picard  tau={tau}  K={K}\n"
                f"Sequential: (N_outer*(K*r+1)) / mean_baseline_iters  |  "
                f"Overlap: N_outer / mean_baseline_iters  |  r=draft/base fwd-pass time",
                fontsize=9, fontweight="bold")

            for row, num_steps in enumerate(steps_l):
                for col, pair in enumerate(pairs_l):
                    ax = axes[row][col]
                    base_name  = pair.split("to")[1]
                    draft_name = pair.split("to")[0]

                    ratio = find_ratio(draft_name, base_name, num_steps, tau)

                    # ---- Baseline: raw Picard iters (integers) ----
                    bkey = f"{base_name}_steps{num_steps}_tau{tau}"
                    if bkey not in baseline:
                        ax.set_visible(False); continue
                    base_iters = [int(r["iters"]) for r in baseline[bkey]]

                    # ---- Sequential depth ----
                    # Each outer iter: K draft calls then 1 base call, fully serial.
                    # depth = N_outer * (K*r + 1), where r = draft_time / base_time.
                    # We show depth as a ratio vs baseline: depth / mean(base_iters).
                    # r is shown in the label so the reader can sanity-check.
                    skey = f"{pair}_steps{num_steps}_tau{tau}_K{K}_sequential"
                    seq_ratios = []
                    if skey in speculative and ratio is not None:
                        cost_seq = K * ratio + 1.0
                        mean_base = float(np.mean(base_iters))
                        for r in speculative[skey]:
                            seq_depth = r["iters"] * cost_seq
                            seq_ratios.append(seq_depth / mean_base if mean_base > 0 else np.nan)

                    # ---- Overlap: just raw outer iters vs baseline iters ----
                    # Base and draft run in parallel; wall time per round ≈ base time
                    # (draft is always cheaper). So outer iters IS the depth in base units.
                    # Compare directly as a ratio vs baseline iters.
                    okey = f"{pair}_steps{num_steps}_tau{tau}_K{K}_overlap"
                    ovl_ratios = []
                    if okey in speculative:
                        mean_base = float(np.mean(base_iters))
                        for r in speculative[okey]:
                            ovl_ratios.append(r["iters"] / mean_base if mean_base > 0 else np.nan)

                    # ---- Plot: side-by-side boxplots of depth ratio ----
                    box_data, labels, colors = [], [], []
                    # Baseline always = ratio 1.0 (show as reference line, not a box)
                    if seq_ratios:
                        r_str = f"r={ratio:.2f}" if ratio is not None else "r=?"
                        box_data.append(seq_ratios)
                        labels.append(f"Sequential\ndepth ratio\n({r_str}, K={K})")
                        colors.append("darkorange")
                    if ovl_ratios:
                        box_data.append(ovl_ratios)
                        labels.append(f"Overlap\niter ratio\n(raw iters / baseline)")
                        colors.append("steelblue")

                    if not box_data:
                        ax.set_visible(False); continue

                    bp = ax.boxplot(box_data, patch_artist=True,
                                    medianprops=dict(color="black", linewidth=1.5))
                    for patch, clr in zip(bp["boxes"], colors):
                        patch.set_facecolor(clr); patch.set_alpha(0.6)

                    # Reference line at 1.0 = same cost as baseline
                    ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
                               label="Baseline (1.0)")
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels, fontsize=7)
                    if col == 0:
                        ax.set_ylabel(f"steps={num_steps}\nRatio vs baseline", fontsize=8)
                    ax.set_title(pair.replace("to", " -> "), fontsize=9)
                    if col == 0 and row == 0:
                        ax.legend(fontsize=7)

            plt.tight_layout()
            path = f"{PLOTS_DIR}/04b_seq_depth_tau{tau}_K{K}.png"
            plt.savefig(path, dpi=150); plt.close()
            print(f"Saved {path}")

# ------------------------------------------------------------------ #
# Fig 5: Residual heatmap — one per pair x overlap_str
#         Uses the highest num_steps available for each pair
# ------------------------------------------------------------------ #
if speculative:
    for pair in spec_pairs():
        # find the largest K and most common tau for this pair
        pair_keys = [k for k in speculative if k.startswith(pair + "_")]
        if not pair_keys:
            continue
        Ks_for_pair   = sorted(set(int(k.split("_K")[1].split("_")[0]) for k in pair_keys))
        taus_for_pair = sorted(set(float(k.split("_tau")[1].split("_")[0]) for k in pair_keys))
        steps_for_pair = sorted(set(int(k.split("_steps")[1].split("_")[0]) for k in pair_keys))

        K_repr    = Ks_for_pair[-1]
        tau_repr  = taus_for_pair[len(taus_for_pair) // 2]   # middle tau
        step_repr = steps_for_pair[-1]

        for overlap_str in ["overlap", "sequential"]:
            key = f"{pair}_steps{step_repr}_tau{tau_repr}_K{K_repr}_{overlap_str}"
            if key not in speculative:
                continue

            # Also need the baseline residual at each picard iteration for this
            # base model / steps / tau, to normalise against.
            base_name = pair.split("to")[1]
            bkey      = f"{base_name}_steps{step_repr}_tau{tau_repr}"

            all_grids = [r["draft_residual_grid"] for r in speculative[key]]
            max_len   = max(len(g) for g in all_grids)
            n_cands   = K_repr + 1

            # Mean absolute residual grid: shape (max_len, n_cands)
            mat    = np.zeros((max_len, n_cands))
            counts = np.zeros(max_len)
            for grid in all_grids:
                for t_idx, row_v in enumerate(grid):
                    for c_idx in range(min(len(row_v), n_cands)):
                        mat[t_idx, c_idx] += row_v[c_idx]
                    counts[t_idx] += 1
            for t_idx in range(max_len):
                if counts[t_idx] > 0:
                    mat[t_idx] /= counts[t_idx]

            # Baseline picard: get mean residual per iteration from residual_history.
            # residual_history is a list of per-step residuals saved at each picard iter.
            # We use the base-only candidate (col 0) as a proxy if no separate baseline
            # history is available, falling back to mat[:, 0].
            base_col = mat[:, 0].copy()   # candidate 0 = base ran alone, no draft

            # Normalise: ratio = candidate_residual / base_residual at same picard iter
            # ratio < 1  → greener (draft helped, lower residual than base alone)
            # ratio > 1  → redder  (draft hurt,  higher residual than base alone)
            # Avoid div-by-zero; clamp display to [0.5, 2.0] so colour is readable
            base_col_safe = np.where(base_col > 0, base_col, np.nan)
            ratio_mat = mat / base_col_safe[:, np.newaxis]   # (max_len, n_cands)
            ratio_mat = np.clip(ratio_mat, 0.5, 2.0)

            # Use a diverging colormap centred at 1.0 (white = same as baseline)
            import matplotlib.colors as mcolors
            norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            im = ax.imshow(ratio_mat.T, aspect="auto", origin="lower",
                           cmap="RdYlGn_r", norm=norm, interpolation="nearest")
            ax.set_xlabel("Picard iteration")
            ax.set_ylabel("Draft candidate (0 = base only)")
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_title(
                f"Residual ratio vs base-only  [{overlap_str}]\n"
                f"{pair.replace('to', ' -> ')}  steps={step_repr}"
                f"  K={K_repr}  tau={tau_repr}\n"
                f"Green = more accurate than base | Red = less accurate"
            )
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("residual / base residual  (1.0 = same as baseline)")
            cbar.ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
            plt.tight_layout()
            path = f"{PLOTS_DIR}/05_residual_heatmap_{pair}_{overlap_str}.png"
            plt.savefig(path, dpi=150); plt.close()
            print(f"Saved {path}")

print("\nAll plots saved to", PLOTS_DIR)