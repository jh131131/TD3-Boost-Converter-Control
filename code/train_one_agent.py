from __future__ import annotations
import argparse
from pathlib import Path
import torch

from boost_td3_ddpg_revised_experiment import train_agent


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one TD3 or DDPG agent for the revised boost-converter experiment.")
    ap.add_argument("algorithm", choices=["TD3", "DDPG"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="revised_experiment")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out) / "training" / f"{args.algorithm.lower()}_seed{args.seed}"
    train_agent(args.algorithm, args.seed, out, args.episodes, args.steps)
    print(f"Saved {args.algorithm} seed {args.seed} to {out}")


if __name__ == "__main__":
    main()
