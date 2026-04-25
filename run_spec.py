from __future__ import annotations

import argparse

import torch

import eval_spec_baseline
import eval_mssim
import eval_spec_spec_traj
import eval_spec_two_picard_grid
import eval_spec_two_picard_iterations

SPEC_RUNNERS = {
    "baseline": eval_spec_baseline.run,
    "mssim": eval_mssim.run,
    "two_picard_iterations": eval_spec_two_picard_iterations.run,
    "speculative": eval_spec_spec_traj.run,
    "two_picard_grid": eval_spec_two_picard_grid.run,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        nargs="+",
        choices=list(SPEC_RUNNERS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_specs = list(SPEC_RUNNERS.keys()) if "all" in args.spec else args.spec

    for spec_name in selected_specs:
        runner = SPEC_RUNNERS[spec_name]
        if spec_name in {"two_picard_grid", "mssim"}:
            runner(force=args.force)
        elif args.num_images is None:
            runner(force=args.force)
        else:
            runner(num_images=args.num_images, force=args.force)


if __name__ == "__main__":
    with torch.no_grad():
        main()
