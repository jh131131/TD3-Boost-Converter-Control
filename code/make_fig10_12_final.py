#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Figure 10-12 from recomputed trajectory CSV files.

Figure 10: Output-voltage response details under representative disturbance scenarios
Figure 11: Duty-cycle trajectories under different disturbance scenarios
Figure 12: Absolute voltage tracking error trajectories under different disturbance scenarios

Usage:
    python make_fig10_12_final.py --input results_recomputed_metrics --out figures_extra_final

Notes:
- This script reads trajectory CSV files produced by recompute_tables_4_6_consistent.py.
- It supports both "param" and "parameter" scenario filename prefixes.
- It does not use the inductor-current plot as CCM verification, avoiding conflict with possible i_L=0 cases.
"""

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS = [
    ("PI", "PI"),
    ("SMC", "SMC"),
    ("DDPG", "DDPG"),
    ("Proposed_TD3", "Proposed TD3"),
]

SCENARIOS = [
    ("load", ["load"], "(a) Load disturbance"),
    ("input", ["input"], "(b) Input voltage fluctuation"),
    ("param", ["param", "parameter"], "(c) Parameter variation"),
]


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_col(df: pd.DataFrame, candidates):
    norm_to_col = {_norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_name(cand)
        if key in norm_to_col:
            return norm_to_col[key]
    # fuzzy matching
    for col in df.columns:
        n = _norm_name(col)
        for cand in candidates:
            c = _norm_name(cand)
            if c and (c in n or n in c):
                return col
    return None


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def find_trajectory_file(input_dir: Path, scenario_aliases, method_key):
    patterns = []
    for alias in scenario_aliases:
        patterns.extend([
            f"trajectory_{alias}_{method_key}.csv",
            f"trajectory_{alias}_{method_key}.CSV",
            f"trajectory_{alias}_{method_key}.txt",
        ])
    for p in patterns:
        file = input_dir / p
        if file.exists():
            return file

    # fallback glob in case extension is hidden/changed
    for alias in scenario_aliases:
        matches = sorted(input_dir.glob(f"trajectory_{alias}_{method_key}*"))
        if matches:
            return matches[0]
    return None


def load_trajectories(input_dir: Path):
    data = {}
    for scenario_key, aliases, title in SCENARIOS:
        data[scenario_key] = {}
        for method_key, method_label in METHODS:
            file = find_trajectory_file(input_dir, aliases, method_key)
            if file is None:
                print(f"[WARN] Missing trajectory file for scenario={scenario_key}, method={method_key}")
                continue

            df = read_csv_safely(file)
            time_col = find_col(df, ["time", "t", "Time", "Time (s)"])
            vo_col = find_col(df, [
                "vo", "v_o", "vout", "v_out", "output_voltage",
                "output voltage", "Output voltage (V)", "voltage"
            ])
            duty_col = find_col(df, [
                "u", "duty", "duty_cycle", "dutycycle",
                "Duty cycle", "control", "action"
            ])

            if time_col is None or vo_col is None:
                print(f"[WARN] Could not identify time/output voltage columns in {file.name}. Columns={list(df.columns)}")
                continue

            item = {
                "file": file,
                "time": pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float),
                "vo": pd.to_numeric(df[vo_col], errors="coerce").to_numpy(dtype=float),
                "duty": None,
            }
            if duty_col is not None:
                item["duty"] = pd.to_numeric(df[duty_col], errors="coerce").to_numpy(dtype=float)
            else:
                print(f"[WARN] Could not identify duty-cycle column in {file.name}; Figure 11 may be incomplete.")

            mask = np.isfinite(item["time"]) & np.isfinite(item["vo"])
            if item["duty"] is not None:
                mask = mask & np.isfinite(item["duty"])
            item["time"] = item["time"][mask]
            item["vo"] = item["vo"][mask]
            if item["duty"] is not None:
                item["duty"] = item["duty"][mask]

            data[scenario_key][method_label] = item
    return data


def set_common_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })


def plot_voltage_details(data, out_dir: Path, vref: float):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)

    # Use local windows for load/input; full window for parameter variation to avoid empty/constant zoom.
    windows = {
        "load": (0.45, 0.80),
        "input": (0.45, 0.80),
        "param": None,
    }

    for ax, (scenario_key, _, title) in zip(axes, SCENARIOS):
        ax.set_title(title)
        win = windows[scenario_key]
        for _, method_label in METHODS:
            item = data.get(scenario_key, {}).get(method_label)
            if item is None:
                continue
            t = item["time"]
            y = item["vo"]
            mask = np.ones_like(t, dtype=bool)
            if win is not None:
                mask = (t >= win[0]) & (t <= win[1])
            ax.plot(t[mask], y[mask], linewidth=1.2, label=method_label)
        ax.axhline(vref, linestyle="--", linewidth=1.0, label=r"$V_{ref}$")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Output voltage (V)")

    handles, labels = axes[0].get_legend_handles_labels()
    # Use unique labels preserving order
    unique = {}
    for h, lab in zip(handles, labels):
        unique[lab] = h
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    fig.savefig(out_dir / "fig10_output_voltage_response_details.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig10_output_voltage_response_details.png", bbox_inches="tight")
    plt.close(fig)


def plot_duty(data, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)

    for ax, (scenario_key, _, title) in zip(axes, SCENARIOS):
        ax.set_title(title)
        for _, method_label in METHODS:
            item = data.get(scenario_key, {}).get(method_label)
            if item is None or item["duty"] is None:
                continue
            ax.plot(item["time"], item["duty"], linewidth=1.2, label=method_label)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Duty cycle")

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, lab in zip(handles, labels):
        unique[lab] = h
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    fig.savefig(out_dir / "fig11_duty_cycle_trajectories.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig11_duty_cycle_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def plot_abs_error(data, out_dir: Path, vref: float):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=False)

    report_lines = []
    for ax, (scenario_key, _, title) in zip(axes, SCENARIOS):
        ax.set_title(title)
        for _, method_label in METHODS:
            item = data.get(scenario_key, {}).get(method_label)
            if item is None:
                continue
            err = np.abs(vref - item["vo"])
            ax.plot(item["time"], err, linewidth=1.2, label=method_label)
            report_lines.append(f"{scenario_key},{method_label},mean_abs_error={np.nanmean(err):.6f},max_abs_error={np.nanmax(err):.6f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absolute tracking error (V)")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, lab in zip(handles, labels):
        unique[lab] = h
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    fig.savefig(out_dir / "fig12_absolute_tracking_error_trajectories.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig12_absolute_tracking_error_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    (out_dir / "fig12_absolute_error_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder containing trajectory CSV files.")
    parser.add_argument("--out", required=True, help="Output folder for figures.")
    parser.add_argument("--vref", type=float, default=200.0, help="Reference output voltage.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input folder does not exist: {input_dir}")
        sys.exit(1)

    set_common_style()
    data = load_trajectories(input_dir)

    plot_voltage_details(data, out_dir, args.vref)
    plot_duty(data, out_dir)
    plot_abs_error(data, out_dir, args.vref)

    print(f"Done. Figures saved to: {out_dir.resolve()}")
    print("Generated:")
    print("  fig10_output_voltage_response_details.pdf/png")
    print("  fig11_duty_cycle_trajectories.pdf/png")
    print("  fig12_absolute_tracking_error_trajectories.pdf/png")
    print("  fig12_absolute_error_report.txt")


if __name__ == "__main__":
    main()
