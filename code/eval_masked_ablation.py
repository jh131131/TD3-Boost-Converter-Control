from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

from boost_td3_ddpg_revised_experiment import BoostParams, BoostConverterEnv, TD3Agent, run_controller, metrics


def load_agent(path: Path) -> TD3Agent:
    p = BoostParams()
    a = TD3Agent(7, 1, p)
    a.actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    a.actor_target.load_state_dict(a.actor.state_dict())
    return a


def mask_state(s: np.ndarray, kind: str) -> np.ndarray:
    s = np.array(s, copy=True)
    if kind in ("no_vin", "no_both"):
        s[4] = 1.0
    if kind in ("no_load", "no_both"):
        s[5] = 1.0
    return s.astype(np.float32)


def run_masked(agent: TD3Agent, kind: str, scenario: str):
    p = BoostParams()
    env = BoostConverterEnv(p, scenario, 4000, seed=999)
    s = mask_state(env.reset(scenario), kind)
    out = {"t": [], "vo": [], "u": [], "iL": [], "bound": []}
    for _ in range(4000):
        u = agent.select_action(s, 0.0)
        ns, _, _, info = env.step(u)
        s = mask_state(ns, kind)
        out["t"].append(env._time())
        out["vo"].append(info["vo"])
        out["u"].append(info["u"])
        out["iL"].append(info["iL"])
        out["bound"].append(info["bound_hit"])
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the observation-signal ablation models.")
    ap.add_argument("--root", default="revised_experiment", help="Experiment output root containing training/ and ablation/.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)

    root = Path(args.root)
    paths = {
        "Full TD3": root / "training" / f"td3_seed{args.seed}" / "td3_actor.pt",
        "w/o Vin": root / "ablation" / f"no_vin_seed{args.seed}" / "td3_actor.pt",
        "w/o Robs": root / "ablation" / f"no_load_seed{args.seed}" / "td3_actor.pt",
        "w/o both disturbance signals": root / "ablation" / f"no_both_seed{args.seed}" / "td3_actor.pt",
    }
    kind_map = {"Full TD3": "full", "w/o Vin": "no_vin", "w/o Robs": "no_load", "w/o both disturbance signals": "no_both"}

    rows = []
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        agent = load_agent(path)
        for sc in ("load", "input", "param"):
            rr = run_controller("TD3", agent, sc) if name == "Full TD3" else run_masked(agent, kind_map[name], sc)
            met = metrics(rr, sc)
            rows.append([name, sc, met["rmse"], met["overshoot"], met["fw_error"]])

    out = Path(args.out) if args.out else root / "observation_ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Scenario", "RMSE (V)", "Overshoot (%)", "Final-window error (V)"])
        w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
