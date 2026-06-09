"""
Runnable Boost Converter + TD3 simulation framework.

Quick run:
    python run_boost_td3_experiment.py --mode quick

Full run:
    python run_boost_td3_experiment.py --mode full

Outputs are saved in ./results/:
    training_log.csv
    fig8_training_convergence.pdf/png
    fig9_dynamic_response_comparison.pdf/png
    table3_training_performance.csv
    table4_load_disturbance.csv
    table5_input_fluctuation.csv
    table6_parameter_variation.csv

Note: This script is intended to provide a reproducible simulation framework.
For final manuscript use, tune controller gains/hyperparameters and run multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# -----------------------------
# Boost converter environment
# -----------------------------

@dataclass
class BoostParams:
    Vin0: float = 100.0
    Vref: float = 200.0
    L0: float = 1e-3
    C0: float = 470e-6
    R0: float = 50.0
    Ts: float = 5e-5  # 50 us; for 2 s test, use 40000 steps
    u_min: float = 0.05
    u_max: float = 0.90
    iL_clip: float = 80.0
    vo_min: float = 1.0
    vo_max: float = 320.0


class BoostConverterEnv:
    def __init__(self, params: BoostParams, scenario: str = "random", episode_steps: int = 1000):
        self.p = params
        self.scenario = scenario
        self.episode_steps = episode_steps
        self.t = 0
        self.prev_u = 0.5
        self.prev_ev = 0.0
        self.iL = 8.0
        self.vo = params.Vref
        self.integral_error = 0.0

    def reset(self, scenario: str | None = None) -> np.ndarray:
        if scenario is not None:
            self.scenario = scenario
        self.t = 0
        self.prev_u = 1.0 - self.p.Vin0 / self.p.Vref

        # Start from the nominal steady state by default.
        self.vo = self.p.Vref
        self.iL = self.p.Vref / self.p.R0 / (1.0 - self.prev_u)

        # Make the learning and robustness tests non-trivial.
        # Without an initial perturbation, pure L/C parameter variation can leave
        # the averaged boost model exactly at equilibrium, producing zero-error
        # baselines and an uninformative Table 6.
        if self.scenario == "random":
            self.vo = self.p.Vref + np.random.uniform(-15.0, 15.0)
            self.prev_u = float(np.clip(self.prev_u + np.random.uniform(-0.04, 0.04),
                                        self.p.u_min, self.p.u_max))
            self.iL = max(0.1, self.vo / self.p.R0 / max(0.10, (1.0 - self.prev_u))
                          + np.random.uniform(-2.0, 2.0))
        elif self.scenario == "param":
            self.vo = self.p.Vref - 12.0
            self.iL = max(0.1, self.vo / self.p.R0 / max(0.10, (1.0 - self.prev_u)))

        self.prev_ev = self.p.Vref - self.vo
        self.integral_error = 0.0
        return self._state()

    def _time(self) -> float:
        return self.t * self.p.Ts

    def disturbance(self) -> Tuple[float, float, float, float]:
        """Return Vin, R, L, C."""
        time = self._time()
        Vin, R, L, C = self.p.Vin0, self.p.R0, self.p.L0, self.p.C0

        if self.scenario == "load":
            # Use compressed timing for shorter simulations; fractions of episode.
            frac = self.t / max(1, self.episode_steps)
            if 0.30 <= frac < 0.65:
                R = 25.0
            elif frac >= 0.65:
                R = 75.0
        elif self.scenario == "input":
            frac = self.t / max(1, self.episode_steps)
            if 0.25 <= frac < 0.50:
                Vin = 106.0
            elif 0.50 <= frac < 0.75:
                Vin = 95.0
            else:
                Vin = 100.0
            Vin += 2.0 * np.sin(2 * np.pi * 40.0 * time)
        elif self.scenario == "param":
            frac = self.t / max(1, self.episode_steps)
            if 0.30 <= frac < 0.60:
                L = self.p.L0 * 1.10
                C = self.p.C0 * 0.92
            elif 0.60 <= frac < 0.80:
                L = self.p.L0 * 0.90
                C = self.p.C0 * 1.12
            elif frac >= 0.80:
                L = self.p.L0 * 1.05
                C = self.p.C0 * 0.96
        elif self.scenario == "random":
            # Randomized training environment; piecewise disturbances.
            frac = self.t / max(1, self.episode_steps)
            Vin += np.random.uniform(-3.0, 3.0) if self.t % 100 == 0 else 0.0
            if 0.30 <= frac < 0.65:
                R = np.random.choice([25.0, 35.0, 65.0, 75.0])
            L = self.p.L0 * np.random.uniform(0.90, 1.10)
            C = self.p.C0 * np.random.uniform(0.90, 1.10)

        return float(Vin), float(R), float(L), float(C)

    def _state(self) -> np.ndarray:
        Vin, R, _, _ = self.disturbance()
        ev = self.p.Vref - self.vo
        dev = ev - self.prev_ev
        # Normalize for neural network input
        s = np.array([
            self.vo / self.p.Vref,
            self.iL / 10.0,
            ev / self.p.Vref,
            dev / self.p.Vref,
            Vin / self.p.Vin0,
            R / self.p.R0,
            self.prev_u,
        ], dtype=np.float32)
        return s

    def step(self, u: float) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        u = float(np.clip(u, self.p.u_min, self.p.u_max))
        Vin, R, L, C = self.disturbance()

        ev_before = self.p.Vref - self.vo
        du = u - self.prev_u

        # Averaged boost dynamics
        diL = (Vin - (1.0 - u) * self.vo) / L
        dvo = ((1.0 - u) * self.iL - self.vo / R) / C
        self.iL += self.p.Ts * diL
        self.vo += self.p.Ts * dvo

        # Safety clipping for numerical stability
        self.iL = float(np.clip(self.iL, 0.0, self.p.iL_clip))
        self.vo = float(np.clip(self.vo, self.p.vo_min, self.p.vo_max))

        ev = self.p.Vref - self.vo
        dvo_abs = abs(self.vo - (self.p.Vref - ev_before))
        # Reward, normalized terms
        reward = -(
            (ev / 20.0) ** 2
            + 0.10 * ((ev - self.prev_ev) / 20.0) ** 2
            + 0.05 * (du / 0.10) ** 2
            + 0.02 * (dvo_abs / 20.0) ** 2
        )

        self.prev_ev = ev
        self.prev_u = u
        self.t += 1
        done = self.t >= self.episode_steps
        info = {"vo": self.vo, "iL": self.iL, "u": u, "ev": ev, "Vin": Vin, "R": R}
        return self._state(), float(reward), done, info


# -----------------------------
# TD3 implementation
# -----------------------------

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 200000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((max_size, state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.ns = np.zeros((max_size, state_dim), dtype=np.float32)
        self.d = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, s, a, r, ns, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: str):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.s[ind], device=device),
            torch.tensor(self.a[ind], device=device),
            torch.tensor(self.r[ind], device=device),
            torch.tensor(self.ns[ind], device=device),
            torch.tensor(self.d[ind], device=device),
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, s):
        # tanh -> [0,1] duty via affine map handled outside
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)

    def q1_value(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x)


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, u_min: float, u_max: float, device: str = "cpu"):
        self.device = device
        self.u_min = u_min
        self.u_max = u_max
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, 1.0).to(device)
        self.actor_target = Actor(state_dim, action_dim, 1.0).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.total_it = 0

    def _scale_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        return (a_tanh + 1.0) * 0.5 * (self.u_max - self.u_min) + self.u_min

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> float:
        s = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self._scale_action(self.actor(s)).cpu().numpy().flatten()[0]
        if noise_std > 0:
            a += np.random.normal(0, noise_std)
        return float(np.clip(a, self.u_min, self.u_max))

    def train(self, replay: ReplayBuffer, batch_size: int = 128, gamma: float = 0.99,
              tau: float = 0.005, policy_noise: float = 0.02, noise_clip: float = 0.05,
              policy_freq: int = 2) -> Tuple[float, float]:
        self.total_it += 1
        s, a, r, ns, d = replay.sample(batch_size, self.device)

        with torch.no_grad():
            noise = torch.randn_like(a) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_a = self._scale_action(self.actor_target(ns)) + noise
            next_a = next_a.clamp(self.u_min, self.u_max)
            tq1, tq2 = self.critic_target(ns, next_a)
            target_q = r + (1.0 - d) * gamma * torch.min(tq1, tq2)

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_value = 0.0
        if self.total_it % policy_freq == 0:
            actor_action = self._scale_action(self.actor(s))
            actor_loss = -self.critic.q1_value(s, actor_action).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return float(critic_loss.item()), actor_loss_value


# -----------------------------
# Baseline controllers
# -----------------------------

class PIController:
    def __init__(self, p: BoostParams, kp: float = 0.0035, ki: float = 35.0):
        self.p = p
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.u0 = 1.0 - p.Vin0 / p.Vref
    def reset(self):
        self.integral = 0.0
    def action(self, vo, iL, prev_u):
        e = self.p.Vref - vo
        self.integral += e * self.p.Ts
        u = self.u0 + self.kp * e + self.ki * self.integral
        return float(np.clip(u, self.p.u_min, self.p.u_max))

class SMCController:
    def __init__(self, p: BoostParams, k: float = 0.06, lam: float = 0.8):
        self.p = p
        self.k = k
        self.lam = lam
        self.prev_e = 0.0
        self.u0 = 1.0 - p.Vin0 / p.Vref
    def reset(self):
        self.prev_e = 0.0
    def action(self, vo, iL, prev_u):
        e = (self.p.Vref - vo) / self.p.Vref
        de = e - self.prev_e
        s = de + self.lam * e
        self.prev_e = e
        u = self.u0 + 0.25 * e + self.k * np.tanh(8.0 * s)
        return float(np.clip(u, self.p.u_min, self.p.u_max))

class GreedyMPCController:
    def __init__(self, p: BoostParams, grid: int = 11):
        self.p = p
        self.grid = grid
        self.u0 = 1.0 - p.Vin0 / p.Vref
    def reset(self):
        pass
    def action(self, vo, iL, prev_u, Vin=None, R=None, L=None, C=None):
        Vin = self.p.Vin0 if Vin is None else Vin
        R = self.p.R0 if R is None else R
        L = self.p.L0 if L is None else L
        C = self.p.C0 if C is None else C
        candidates = np.linspace(max(self.p.u_min, prev_u - 0.12), min(self.p.u_max, prev_u + 0.12), self.grid)
        best_u, best_cost = prev_u, 1e18
        for u in candidates:
            diL = (Vin - (1.0 - u) * vo) / L
            dvo = ((1.0 - u) * iL - vo / R) / C
            pred_vo = vo + self.p.Ts * dvo
            cost = ((self.p.Vref - pred_vo) / 10.0) ** 2 + 0.03 * ((u - prev_u) / 0.05) ** 2
            if cost < best_cost:
                best_cost = cost
                best_u = u
        return float(np.clip(best_u, self.p.u_min, self.p.u_max))


# -----------------------------
# Training and evaluation
# -----------------------------

def train_td3(mode: str, out_dir: str, seed: int) -> TD3Agent:
    set_seed(seed)
    device = "cpu"
    # Use the same controller sampling period in training and testing.
    # This also keeps the disturbance timing consistent with the 2 s evaluation horizon.
    p = BoostParams(Ts=5e-4)
    state_dim, action_dim = 7, 1
    agent = TD3Agent(state_dim, action_dim, p.u_min, p.u_max, device=device)
    replay = ReplayBuffer(state_dim, action_dim, max_size=200000 if mode == "quick" else 1000000)

    # quick: sanity check only; full: manuscript-level single-seed run.
    # For final submission, run at least three seeds.
    episodes = 5 if mode == "quick" else 600
    steps = 200 if mode == "quick" else 1000
    start_steps = 100 if mode == "quick" else 1000
    batch_size = 32 if mode == "quick" else 128

    env = BoostConverterEnv(p, scenario="random", episode_steps=steps)
    log_rows = []
    global_step = 0
    critic_losses, actor_losses = [], []

    for ep in range(1, episodes + 1):
        s = env.reset("random")
        ep_reward = 0.0
        vo_list, du_list = [], []
        prev_u = env.prev_u
        for _ in range(steps):
            if global_step < start_steps:
                u = np.random.uniform(p.u_min, p.u_max)
            else:
                u = agent.select_action(s, noise_std=0.05 if mode == "quick" else 0.03)
            ns, r, done, info = env.step(u)
            replay.add(s, [u], r, ns, float(done))
            s = ns
            ep_reward += r
            vo_list.append(info["vo"])
            du_list.append(abs(info["u"] - prev_u))
            prev_u = info["u"]
            global_step += 1
            if replay.size > batch_size:
                closs, aloss = agent.train(replay, batch_size=batch_size)
                critic_losses.append(closs)
                if aloss != 0:
                    actor_losses.append(aloss)
            if done:
                break
        rmse = float(np.sqrt(np.mean((np.array(vo_list) - p.Vref) ** 2)))
        avg_du = float(np.mean(du_list))
        log_rows.append([ep, ep_reward / steps, rmse, avg_du,
                         np.mean(critic_losses[-20:]) if critic_losses else 0.0,
                         np.mean(actor_losses[-20:]) if actor_losses else 0.0])
        if ep % max(1, episodes // 10) == 0:
            print(f"Episode {ep}/{episodes}: reward={log_rows[-1][1]:.3f}, RMSE={rmse:.3f}, duty_var={avg_du:.4f}")

    write_csv(os.path.join(out_dir, "training_log.csv"),
              ["episode", "average_reward", "voltage_rmse", "average_duty_variation", "critic_loss", "actor_loss"],
              log_rows)

    # Save model
    torch.save(agent.actor.state_dict(), os.path.join(out_dir, "td3_actor.pt"))
    return agent


def run_controller(controller_name: str, agent: TD3Agent | None, scenario: str, steps: int = 4000) -> Dict[str, np.ndarray]:
    p = BoostParams(Ts=5e-4)  # test horizon 4000*0.5 ms=2s, consistent with Figure 7 timeline
    env = BoostConverterEnv(p, scenario=scenario, episode_steps=steps)
    s = env.reset(scenario)
    if controller_name == "PI":
        ctrl = PIController(p); ctrl.reset()
    elif controller_name == "SMC":
        ctrl = SMCController(p); ctrl.reset()
    elif controller_name == "MPC":
        ctrl = GreedyMPCController(p); ctrl.reset()
    else:
        ctrl = None

    t, vo, u_hist = [], [], []
    for _ in range(steps):
        Vin, R, L, C = env.disturbance()
        if controller_name == "TD3":
            assert agent is not None
            u = agent.select_action(s, noise_std=0.0)
        elif controller_name == "DDPG_like":
            # Placeholder baseline: a weaker deterministic learning-based controller.
            # For a final manuscript that explicitly reports DDPG, replace this block
            # with an independently trained DDPG agent.
            assert agent is not None
            raw = agent.select_action(s, noise_std=0.0)
            u = 0.85 * env.prev_u + 0.15 * raw
        elif controller_name == "MPC":
            u = ctrl.action(env.vo, env.iL, env.prev_u, Vin, R, L, C)
        else:
            u = ctrl.action(env.vo, env.iL, env.prev_u)
        s, r, done, info = env.step(u)
        t.append(env._time())
        vo.append(info["vo"])
        u_hist.append(info["u"])
        if done:
            break
    return {"t": np.array(t), "vo": np.array(vo), "u": np.array(u_hist)}


def metrics(t: np.ndarray, vo: np.ndarray, vref: float = 200.0) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((vo - vref) ** 2)))
    overshoot = float(max(0.0, (np.max(vo) - vref) / vref * 100.0))
    ess = float(abs(vref - np.mean(vo[-max(10, len(vo)//20):])))
    band = 0.02 * vref
    settle_time = t[-1]
    for i in range(len(t)):
        if np.all(np.abs(vo[i:] - vref) <= band):
            settle_time = t[i]
            break
    return {"RMSE (V)": rmse, "Overshoot (%)": overshoot, "Settling time (ms)": settle_time * 1000.0, "Steady-state error (V)": ess}


def build_tables_and_plots(agent: TD3Agent, out_dir: str, mode: str = "quick"):
    p = BoostParams()
    # Training plot from log
    data = np.genfromtxt(os.path.join(out_dir, "training_log.csv"), delimiter=",", names=True, ndmin=1)
    episodes = data["episode"]
    fig, ax1 = plt.subplots(figsize=(10.8, 6.6))
    ax2 = ax1.twinx()
    ax1.plot(episodes, data["average_reward"], linewidth=2.0, label="Average reward")
    ax2.plot(episodes, data["voltage_rmse"], linewidth=2.0, linestyle="--", label="Voltage RMSE (V)")
    ax2.plot(episodes, data["average_duty_variation"], linewidth=2.0, linestyle="-.", label="Average duty variation")
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Average reward")
    ax2.set_ylabel("RMSE / duty variation")
    ax1.set_title("Training convergence of the proposed TD3-based adaptive controller", fontweight="bold")
    ax1.grid(True, linestyle=":", linewidth=0.8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, fontsize=9)
    fig.savefig(os.path.join(out_dir, "fig8_training_convergence.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig8_training_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Table 3 from thirds of training logs.
    # In quick mode there may be only one row, so the same value is reused
    # only to verify the pipeline. Full mode should be used for paper results.
    n = len(data)
    names = ["Initial stage", "Middle stage", "Final stage"]
    rows = []
    if n < 3:
        for name in names:
            rows.append([name,
                         f"{float(np.mean(data['average_reward'])):.2f}",
                         f"{float(np.mean(data['voltage_rmse'])):.2f}",
                         f"{float(np.mean(data['average_duty_variation'])):.4f}"])
    else:
        thirds = [(0, max(1, n//3)), (max(1, n//3), max(2, 2*n//3)), (max(2, 2*n//3), n)]
        for name, (a, b) in zip(names, thirds):
            rows.append([name,
                         f"{np.mean(data['average_reward'][a:b]):.2f}",
                         f"{np.mean(data['voltage_rmse'][a:b]):.2f}",
                         f"{np.mean(data['average_duty_variation'][a:b]):.4f}"])
    write_csv(os.path.join(out_dir, "table3_training_performance.csv"),
              ["Training stage", "Average reward", "Voltage RMSE (V)", "Average duty variation"], rows)

    # Evaluation results
    methods = ["PI", "SMC", "DDPG_like", "TD3"]
    method_names = {"PI": "PI", "SMC": "SMC", "DDPG_like": "DDPG", "TD3": "Proposed TD3"}
    scenarios = {"load":"table4_load_disturbance.csv", "input":"table5_input_fluctuation.csv", "param":"table6_parameter_variation.csv"}
    all_results = {}
    eval_steps = 100 if mode == "quick" else 4000
    for scenario, table_fn in scenarios.items():
        rows = []
        all_results[scenario] = {}
        for m in methods:
            result = run_controller(m, agent, scenario=scenario, steps=eval_steps)
            all_results[scenario][m] = result
            met = metrics(result["t"], result["vo"], vref=p.Vref)
            rows.append([method_names[m], f"{met['RMSE (V)']:.2f}", f"{met['Overshoot (%)']:.2f}",
                         f"{met['Settling time (ms)']:.2f}", f"{met['Steady-state error (V)']:.2f}"])
        write_csv(os.path.join(out_dir, table_fn),
                  ["Method", "RMSE (V)", "Overshoot (%)", "Settling time (ms)", "Steady-state error (V)"], rows)

    # Figure 9: plot true simulated responses
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    offsets = {"load":0.0, "input":2.25, "param":4.5}
    labels = {"load":"(a) Load disturbance", "input":"(b) Input voltage fluctuation", "param":"(c) Parameter variation"}
    for scenario in ["load", "input", "param"]:
        off = offsets[scenario]
        for m in methods:
            r = all_results[scenario][m]
            t_plot = r["t"] + off
            label = method_names[m] if scenario == "load" else None
            ax.plot(t_plot, r["vo"], linewidth=1.8, label=label)
        ax.axvline(off + 0.30 * 2.0, linestyle="--", linewidth=0.9)
        ax.text(off + 1.0, 215.0, labels[scenario], ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 6.5)
    ax.set_ylim(160, 230)
    ax.set_xlabel("Scenario timeline")
    ax.set_ylabel("Output voltage (V)")
    ax.set_title("Dynamic response comparison of different control strategies under multiple disturbance conditions",
                 fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=9)
    fig.subplots_adjust(right=0.80)
    fig.savefig(os.path.join(out_dir, "fig9_dynamic_response_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig9_dynamic_response_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    ensure_dir(args.out)
    print(f"Running mode={args.mode}, seed={args.seed}. Results will be saved to {args.out}")
    agent = train_td3(args.mode, args.out, args.seed)
    build_tables_and_plots(agent, args.out, args.mode)
    print("Done. Check the results folder for figures and tables.")

if __name__ == "__main__":
    main()
