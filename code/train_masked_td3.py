from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from boost_td3_ddpg_revised_experiment import (
    BoostParams, BoostConverterEnv, ReplayBuffer, TD3Agent, set_seed, write_csv
)

MASKS = {
    "no_vin": (4,),
    "no_load": (5,),
    "no_both": (4, 5),
}


def mask_state(s: np.ndarray, kind: str) -> np.ndarray:
    s = np.array(s, copy=True)
    if 4 in MASKS[kind]:
        s[4] = 1.0
    if 5 in MASKS[kind]:
        s[5] = 1.0
    return s.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train TD3 observation-signal ablations.")
    ap.add_argument("variant", choices=sorted(MASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="revised_experiment")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    torch.set_num_threads(1)
    set_seed(args.seed)
    p = BoostParams()
    agent = TD3Agent(7, 1, p)
    replay = ReplayBuffer(7, 1, max_size=250000)
    env = BoostConverterEnv(p, "random", args.steps, seed=args.seed + 1234)
    warmup, batch, global_step = 1000, 128, 0
    rows = []

    for ep in range(1, args.episodes + 1):
        s = mask_state(env.reset("random"), args.variant)
        episode_reward, vos, dus = 0.0, [], []
        prev_u = env.prev_u
        critic_losses, actor_losses = [], []
        for _ in range(args.steps):
            u = np.random.uniform(p.u_min, p.u_max) if global_step < warmup else agent.select_action(s, 0.03)
            ns, r, done, info = env.step(u)
            ns = mask_state(ns, args.variant)
            replay.add(s, [u], r, ns, float(done))
            s = ns
            episode_reward += r
            vos.append(info["vo"])
            dus.append(abs(info["u"] - prev_u))
            prev_u = info["u"]
            global_step += 1
            if replay.size > batch:
                cl, al = agent.train(replay, batch)
                critic_losses.append(cl)
                actor_losses.append(al)
            if done:
                break
        rmse = float(np.sqrt(np.mean((np.asarray(vos) - p.Vref) ** 2)))
        rows.append([
            ep,
            episode_reward / args.steps,
            rmse,
            float(np.mean(dus)),
            float(np.mean(critic_losses[-50:])) if critic_losses else 0.0,
            float(np.mean(actor_losses[-50:])) if actor_losses else 0.0,
        ])
        if ep % 5 == 0:
            print(args.variant, ep, rmse, flush=True)

    out = Path(args.out) / "ablation" / f"{args.variant}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "training_log.csv", [
        "episode", "average_reward", "voltage_rmse", "average_duty_variation", "critic_loss", "actor_loss"
    ], rows)
    torch.save(agent.actor.state_dict(), out / "td3_actor.pt")
    print(f"Saved ablation model to {out}")


if __name__ == "__main__":
    main()
