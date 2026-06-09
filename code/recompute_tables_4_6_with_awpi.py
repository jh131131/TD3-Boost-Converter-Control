#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recompute Table 4--Table 6 metrics with an additional Anti-windup PI baseline.

This script does NOT retrain TD3. It only reruns deterministic test trajectories.
It adds one baseline:
    PI-AW: PI controller with conditional anti-windup under duty-cycle saturation.

Put this file in the same folder as run_boost_td3_experiment_revised.py, then run:

    python recompute_tables_4_6_with_awpi.py --module run_boost_td3_experiment_revised --actor results_no_mpc/td3_actor.pt --out results_recomputed_metrics_with_awpi

Important:
- Use the SAME TD3 actor that produced your current manuscript Table 4--6.
- If your current TD3 actor is results_seed_0/td3_actor.pt, replace the --actor path accordingly.

Outputs:
    table4_load_disturbance_recomputed.csv
    table5_input_fluctuation_recomputed.csv
    table6_parameter_variation_recomputed.csv
    trajectory_*.csv files for PI, PI-AW, SMC, DDPG, and Proposed TD3.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class AntiWindupPIController:
    """PI controller with conditional anti-windup.

    The proportional term is always applied. The integral state is updated only
    when the unconstrained duty command is not saturated, or when the current
    error tends to drive the command back from the saturated boundary.
    """

    def __init__(self, p, kp: float = 0.0035, ki: float = 35.0):
        self.p = p
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.u0 = 1.0 - p.Vin0 / p.Vref

    def reset(self):
        self.integral = 0.0

    def action(self, vo, iL, prev_u):
        e = self.p.Vref - vo

        # Candidate integral update.
        integral_candidate = self.integral + e * self.p.Ts
        u_unsat = self.u0 + self.kp * e + self.ki * integral_candidate
        u_sat = float(np.clip(u_unsat, self.p.u_min, self.p.u_max))

        # Conditional integration:
        # 1) no saturation -> accept integration;
        # 2) upper saturation and e < 0 -> accept because it pulls duty downward;
        # 3) lower saturation and e > 0 -> accept because it pulls duty upward.
        no_saturation = (self.p.u_min < u_unsat < self.p.u_max)
        helps_leave_upper = (u_unsat >= self.p.u_max and e < 0)
        helps_leave_lower = (u_unsat <= self.p.u_min and e > 0)

        if no_saturation or helps_leave_upper or helps_leave_lower:
            self.integral = integral_candidate

        u = self.u0 + self.kp * e + self.ki * self.integral
        return float(np.clip(u, self.p.u_min, self.p.u_max))


def recovery_time_ms(t: np.ndarray, vo: np.ndarray, vref: float, start_time: float, band_ratio: float) -> Optional[float]:
    band = band_ratio * vref
    start_idx = int(np.searchsorted(t, start_time, side="left"))
    if start_idx >= len(t):
        return None
    err = np.abs(vo - vref)
    for i in range(start_idx, len(t)):
        if np.all(err[i:] <= band):
            return float((t[i] - start_time) * 1000.0)
    return None


def compute_metrics(
    t: np.ndarray,
    vo: np.ndarray,
    vref: float,
    rmse_start_time: float,
    recovery_start_time: float,
    final_window_s: float,
    band_ratio: float,
    vo_clip: float = 320.0,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    idx_eval = int(np.searchsorted(t, rmse_start_time, side="left"))
    idx_eval = min(max(idx_eval, 0), len(t) - 1)
    vo_eval = vo[idx_eval:]

    err_eval = vo_eval - vref
    rmse = float(np.sqrt(np.mean(err_eval ** 2)))
    overshoot = float(max(0.0, (np.max(vo_eval) - vref) / vref * 100.0))

    t_end = float(t[-1])
    idx_final = int(np.searchsorted(t, max(0.0, t_end - final_window_s), side="left"))
    idx_final = min(max(idx_final, 0), len(t) - 1)
    final_window_error = float(np.mean(np.abs(vo[idx_final:] - vref)))

    rec = recovery_time_ms(t, vo, vref, recovery_start_time, band_ratio)
    rec_text = f"{rec:.1f}" if rec is not None else f">{t_end * 1000:.0f}"

    saturation_limited = bool(np.max(vo_eval) >= (vo_clip - 1e-6))
    diagnostics = {
        "max_vo_after_disturbance": float(np.max(vo_eval)),
        "min_vo_after_disturbance": float(np.min(vo_eval)),
        "saturation_limited_peak": saturation_limited,
        "rmse_start_time_s": rmse_start_time,
        "recovery_start_time_s": recovery_start_time,
        "band_ratio": band_ratio,
        "final_window_s": final_window_s,
    }
    metrics = {
        "RMSE (V)": rmse,
        "Overshoot (%)": overshoot,
        "Recovery time (ms)": rec_text,
        "Final-window error (V)": final_window_error,
    }
    return metrics, diagnostics


def run_controller_full(mod, controller_name: str, agent, scenario: str, steps: int = 4000) -> Dict[str, np.ndarray]:
    p = mod.BoostParams(Ts=5e-4)
    env = mod.BoostConverterEnv(p, scenario=scenario, episode_steps=steps)
    s = env.reset(scenario)

    if controller_name == "PI":
        ctrl = mod.PIController(p); ctrl.reset()
    elif controller_name == "PI_AW":
        ctrl = AntiWindupPIController(p); ctrl.reset()
    elif controller_name == "SMC":
        ctrl = mod.SMCController(p); ctrl.reset()
    else:
        ctrl = None

    rows = {"t": [], "vo": [], "iL": [], "u": [], "ev": [], "Vin": [], "R": [], "L": [], "C": []}
    for _ in range(steps):
        Vin, R, L, C = env.disturbance()
        if controller_name == "TD3":
            u = agent.select_action(s, noise_std=0.0)
        elif controller_name == "DDPG_like":
            raw = agent.select_action(s, noise_std=0.0)
            u = 0.85 * env.prev_u + 0.15 * raw
        else:
            u = ctrl.action(env.vo, env.iL, env.prev_u)

        s, _, done, info = env.step(u)
        rows["t"].append(env._time())
        rows["vo"].append(info["vo"])
        rows["iL"].append(info["iL"])
        rows["u"].append(info["u"])
        rows["ev"].append(info["ev"])
        rows["Vin"].append(info["Vin"])
        rows["R"].append(info["R"])
        rows["L"].append(L)
        rows["C"].append(C)
        if done:
            break

    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="run_boost_td3_experiment_revised",
                    help="Python module name of your simulation script, without .py")
    ap.add_argument("--actor", default="results_no_mpc/td3_actor.pt",
                    help="Path to trained TD3 actor weights")
    ap.add_argument("--out", default="results_recomputed_metrics_with_awpi")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--final_window", type=float, default=0.20)
    args = ap.parse_args()

    ensure_dir(args.out)
    mod = importlib.import_module(args.module)
    p = mod.BoostParams(Ts=5e-4)

    agent = mod.TD3Agent(state_dim=7, action_dim=1, u_min=p.u_min, u_max=p.u_max, device="cpu")
    state_dict = torch.load(args.actor, map_location="cpu")
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()

    methods = ["PI", "PI_AW", "SMC", "DDPG_like", "TD3"]
    method_names = {
        "PI": "PI",
        "PI_AW": "PI-AW",
        "SMC": "SMC",
        "DDPG_like": "DDPG",
        "TD3": "Proposed TD3",
    }

    scenario_cfg = {
        "load":  {"file": "table4_load_disturbance_recomputed.csv",   "rmse_start": 0.60, "recovery_start": 1.30},
        "input": {"file": "table5_input_fluctuation_recomputed.csv", "rmse_start": 0.50, "recovery_start": 1.50},
        "param": {"file": "table6_parameter_variation_recomputed.csv","rmse_start": 0.60, "recovery_start": 1.60},
    }

    diag_rows = []
    for scenario, cfg in scenario_cfg.items():
        table_rows = []
        for m in methods:
            tr = run_controller_full(mod, m, agent, scenario=scenario, steps=args.steps)

            safe_name = method_names[m].replace(" ", "_").replace("-", "_")
            traj_path = os.path.join(args.out, f"trajectory_{scenario}_{safe_name}.csv")
            write_csv(
                traj_path,
                ["time_s", "output_voltage_V", "inductor_current_A", "duty_cycle", "tracking_error_V", "Vin_V", "R_ohm", "L_H", "C_F"],
                zip(tr["t"], tr["vo"], tr["iL"], tr["u"], tr["ev"], tr["Vin"], tr["R"], tr["L"], tr["C"]),
            )

            met, diag = compute_metrics(
                tr["t"], tr["vo"], p.Vref,
                rmse_start_time=cfg["rmse_start"],
                recovery_start_time=cfg["recovery_start"],
                final_window_s=args.final_window,
                band_ratio=args.band,
                vo_clip=p.vo_max,
            )

            table_rows.append([
                method_names[m],
                f"{met['RMSE (V)']:.2f}",
                f"{met['Overshoot (%)']:.2f}",
                met["Recovery time (ms)"],
                f"{met['Final-window error (V)']:.2f}",
            ])

            diag_rows.append([
                scenario, method_names[m],
                f"{diag['max_vo_after_disturbance']:.2f}",
                f"{diag['min_vo_after_disturbance']:.2f}",
                diag["saturation_limited_peak"],
                diag["rmse_start_time_s"],
                diag["recovery_start_time_s"],
                diag["band_ratio"],
                diag["final_window_s"],
            ])

        write_csv(
            os.path.join(args.out, cfg["file"]),
            ["Method", "Post-disturbance RMSE (V)", "Overshoot (%)", "Recovery time (ms)", "Final-window error (V)"],
            table_rows,
        )

    write_csv(
        os.path.join(args.out, "diagnostics_saturation_and_metric_windows.csv"),
        ["Scenario", "Method", "Max vo after disturbance (V)", "Min vo after disturbance (V)",
         "Saturation-limited peak", "RMSE start time (s)", "Recovery start time (s)", "Band ratio", "Final window (s)"],
        diag_rows,
    )

    print(f"Done. Recomputed tables and trajectories saved to: {args.out}")
    print("Generated methods: PI, PI-AW, SMC, DDPG, Proposed TD3")


if __name__ == "__main__":
    main()
