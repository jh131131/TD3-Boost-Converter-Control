#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized Figure 12 generator for the boost TD3 manuscript.

Purpose:
- Generate a clearer absolute tracking error figure for the manuscript.
- Main output uses a zoomed y-axis to show differences among SMC, DDPG, and Proposed TD3.
- Auxiliary full-y-axis version is also generated for checking large PI peaks.
- A report CSV records mean, maximum, and final absolute tracking errors.

Usage:
    python make_fig12_error_optimized.py --input results_recomputed_metrics --out figures_extra_final

Optional:
    python make_fig12_error_optimized.py --input results_recomputed_metrics --out figures_extra_final --vref 200 --zoom_ylim 30
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
    for alias in scenario_aliases:
        for suffix in [".csv", ".CSV", ".txt"]:
            file = input_dir / f"trajectory_{alias}_{method_key}{suffix}"
            if file.exists():
                return file

    for alias in scenario_aliases:
        matches = sorted(input_dir.glob(f"trajectory_{alias}_{method_key}*"))
        if matches:
            return matches[0]
    return None


def load_trajectories(input_dir: Path):
    data = {}
    for scenario_key, aliases, _ in SCENARIOS:
        data[scenario_key] = {}
        for method_key, method_label in METHODS:
            file = find_trajectory_file(input_dir, aliases, method_key)
            if file is None:
                print(f"[WARN] Missing trajectory file: scenario={scenario_key}, method={method_key}")
                continue

            df = read_csv_safely(file)
            time_col = find_col(df, ["time", "t", "Time", "Time (s)"])
            vo_col = find_col(df, [
                "vo", "v_o", "vout", "v_out", "output_voltage",
                "output voltage", "Output voltage (V)", "voltage"
            ])

            if time_col is None or vo_col is None:
                print(f"[WARN] Failed to identify columns in {file.name}. Columns={list(df.columns)}")
                continue

            time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
            vo = pd.to_numeric(df[vo_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(time) & np.isfinite(vo)

            time = time[mask]
            vo = vo[mask]

            if len(time) == 0:
                print(f"[WARN] Empty valid data in {file.name}")
                continue

            data[scenario_key][method_label] = {
                "file": file,
                "time": time,
                "vo": vo,
            }

    return data


def set_style():
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
        "lines.linewidth": 1.25,
    })


def collect_error_stats(data, vref):
    rows = []
    for scenario_key, _, _ in SCENARIOS:
        for _, method_label in METHODS:
            item = data.get(scenario_key, {}).get(method_label)
            if item is None:
                continue
            err = np.abs(vref - item["vo"])
            rows.append({
                "scenario": scenario_key,
                "method": method_label,
                "mean_abs_error": float(np.nanmean(err)),
                "max_abs_error": float(np.nanmax(err)),
                "final_abs_error": float(err[-1]),
                "file": str(item["file"]),
            })
    return rows


def plot_abs_error(data, out_dir: Path, vref: float, zoom_ylim: float, full_scale: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=not full_scale)

    for ax, (scenario_key, _, title) in zip(axes, SCENARIOS):
        ax.set_title(title)
        scenario_max = 0.0

        for _, method_label in METHODS:
            item = data.get(scenario_key, {}).get(method_label)
            if item is None:
                continue
            t = item["time"]
            err = np.abs(vref - item["vo"])
            scenario_max = max(scenario_max, float(np.nanmax(err)))
            ax.plot(t, err, label=method_label)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Absolute tracking error (V)")
        ax.grid(True, alpha=0.25)

        if full_scale:
            ymax = max(zoom_ylim, scenario_max * 1.08)
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0, zoom_ylim)
            if scenario_max > zoom_ylim:
                ax.text(
                    0.03, 0.93,
                    f"peaks exceed {zoom_ylim:g} V",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
                )

    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, lab in zip(handles, labels):
        if lab not in unique:
            unique[lab] = h

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    stem = "fig12_absolute_tracking_error_trajectories_full" if full_scale else "fig12_absolute_tracking_error_trajectories_zoomed"
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder containing trajectory CSV files.")
    parser.add_argument("--out", required=True, help="Output folder for optimized Figure 12.")
    parser.add_argument("--vref", type=float, default=200.0, help="Reference output voltage.")
    parser.add_argument("--zoom_ylim", type=float, default=30.0, help="Y-axis upper limit for zoomed figure.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input folder does not exist: {input_dir}")
        sys.exit(1)

    set_style()
    data = load_trajectories(input_dir)

    plot_abs_error(data, out_dir, args.vref, args.zoom_ylim, full_scale=False)
    plot_abs_error(data, out_dir, args.vref, args.zoom_ylim, full_scale=True)

    rows = collect_error_stats(data, args.vref)
    report_path = out_dir / "fig12_absolute_error_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"Done. Optimized Figure 12 files saved to: {out_dir.resolve()}")
    print("Generated:")
    print("  fig12_absolute_tracking_error_trajectories_zoomed.pdf/png")
    print("  fig12_absolute_tracking_error_trajectories_full.pdf/png")
    print("  fig12_absolute_error_report.csv")
    print()
    print("Recommended manuscript figure: fig12_absolute_tracking_error_trajectories_zoomed.pdf")
    print("Use the full version only for internal checking or supplementary material.")


if __name__ == "__main__":
    main()
