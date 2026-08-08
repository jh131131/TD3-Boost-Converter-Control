from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def run(cmd):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the revised TD3/DDPG experiment and observation ablations.")
    ap.add_argument("--out", default="revised_experiment")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    args = ap.parse_args()
    here = Path(__file__).resolve().parent

    run([sys.executable, str(here / "boost_td3_ddpg_revised_experiment.py"), "--out", args.out,
         "--episodes", str(args.episodes), "--steps", str(args.steps), "--seeds", *map(str, args.seeds)])

    for variant in ("no_vin", "no_load", "no_both"):
        run([sys.executable, str(here / "train_masked_td3.py"), variant, "--seed", "0", "--out", args.out,
             "--episodes", str(args.episodes), "--steps", str(args.steps)])
    run([sys.executable, str(here / "eval_masked_ablation.py"), "--root", args.out, "--seed", "0"])


if __name__ == "__main__":
    main()
