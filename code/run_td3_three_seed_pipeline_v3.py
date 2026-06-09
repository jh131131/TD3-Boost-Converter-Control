#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-seed TD3 experiment pipeline, v3.

This version does NOT patch/copy the training script. It runs your training script
the same way you normally run the full experiment:

    python run_boost_td3_experiment_revised.py --mode full --seed <seed> --out results_seed_<seed>

Then it recomputes Table 4-6 metrics for each trained actor:

    python recompute_tables_4_6_consistent.py --module run_boost_td3_experiment_revised --actor results_seed_<seed>/td3_actor.pt --out three_seed_results_v3/metrics/seed<seed>

Finally, it generates:
    three_seed_results_v3/td3_three_seed_raw_results.csv
    three_seed_results_v3/td3_three_seed_summary.csv
    three_seed_results_v3/table8_td3_seed_statistics_latex.txt

Usage:
    cd /d E:\Desktop\boost_td3_project
    python run_td3_three_seed_pipeline_v3.py --source run_boost_td3_experiment_revised.py --recompute recompute_tables_4_6_consistent.py --seeds 0 1 2

If your full training mode is not called "full", change --mode.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Tuple


SCENARIO_TABLES = {
    "Load disturbance": "table4_load_disturbance_recomputed.csv",
    "Input voltage fluctuation": "table5_input_fluctuation_recomputed.csv",
    "Parameter variation": "table6_parameter_variation_recomputed.csv",
}

TD3_NAMES = {
    "Proposed TD3",
    "TD3",
    "Proposed_TD3",
    "proposed td3",
    "proposed_td3",
}


def run_command(cmd: List[str], cwd: Path, log_file: Path) -> int:
    print("[RUN]", " ".join(str(x) for x in cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", errors="ignore") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        return process.wait()


def read_csv_dicts(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_column(row: dict, candidates: List[str]) -> Optional[str]:
    norm = {
        str(k).strip().lower().replace(" ", "").replace("_", "").replace("-", ""): k
        for k in row.keys()
    }
    for cand in candidates:
        c = cand.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if c in norm:
            return norm[c]
    for k in row.keys():
        kk = str(k).strip().lower()
        for cand in candidates:
            if cand.lower() in kk:
                return k
    return None


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"-", "--", "—"}:
        return None
    s = s.replace("%", "").replace(">", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_td3_metrics(metrics_dir: Path, seed: int) -> List[dict]:
    rows_out = []
    for scenario, filename in SCENARIO_TABLES.items():
        table_path = metrics_dir / filename
        if not table_path.exists():
            raise FileNotFoundError(f"Missing recomputed table: {table_path}")

        rows = read_csv_dicts(table_path)
        if not rows:
            raise RuntimeError(f"Empty CSV: {table_path}")

        method_col = find_column(rows[0], ["Method", "method"])
        if method_col is None:
            raise RuntimeError(f"Could not find Method column in {table_path}")

        td3_row = None
        for r in rows:
            name = str(r.get(method_col, "")).strip()
            if name in TD3_NAMES or name.lower().replace(" ", "_") in TD3_NAMES:
                td3_row = r
                break

        if td3_row is None:
            raise RuntimeError(f"Could not find Proposed TD3 row in {table_path}")

        rmse_col = find_column(td3_row, ["Post-disturbance RMSE (V)", "RMSE (V)", "RMSE"])
        overshoot_col = find_column(td3_row, ["Overshoot (%)", "Overshoot"])
        recovery_col = find_column(td3_row, ["Recovery time (ms)", "Recovery Time", "Settling time"])
        fw_col = find_column(td3_row, ["Final-window error (V)", "Final window error", "Final-window error"])

        rows_out.append({
            "seed": seed,
            "scenario": scenario,
            "rmse": parse_float(td3_row.get(rmse_col)) if rmse_col else None,
            "overshoot": parse_float(td3_row.get(overshoot_col)) if overshoot_col else None,
            "recovery_time": parse_float(td3_row.get(recovery_col)) if recovery_col else None,
            "final_window_error": parse_float(td3_row.get(fw_col)) if fw_col else None,
        })
    return rows_out


def mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, var ** 0.5


def fmt_mean_std(mean: Optional[float], std: Optional[float], digits: int = 2) -> str:
    if mean is None or std is None:
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_summary(results: List[dict], out_dir: Path) -> None:
    raw_path = out_dir / "td3_three_seed_raw_results.csv"
    with raw_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "scenario", "rmse", "overshoot", "recovery_time", "final_window_error"],
        )
        writer.writeheader()
        writer.writerows(results)

    summary_rows = []
    for scenario in SCENARIO_TABLES.keys():
        subset = [r for r in results if r["scenario"] == scenario]
        rmse_m, rmse_s = mean_std([r["rmse"] for r in subset])
        over_m, over_s = mean_std([r["overshoot"] for r in subset])
        rec_m, rec_s = mean_std([r["recovery_time"] for r in subset])
        fw_m, fw_s = mean_std([r["final_window_error"] for r in subset])
        summary_rows.append({
            "scenario": scenario,
            "rmse_mean": rmse_m,
            "rmse_std": rmse_s,
            "overshoot_mean": over_m,
            "overshoot_std": over_s,
            "recovery_time_mean": rec_m,
            "recovery_time_std": rec_s,
            "final_window_error_mean": fw_m,
            "final_window_error_std": fw_s,
        })

    summary_path = out_dir / "td3_three_seed_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical performance of the proposed TD3 controller under three random seeds.}",
        r"\label{tab:td3_seed_statistics}",
        r"\begin{tabular}{llll}",
        r"\hline",
        r"Scenario & RMSE (V) & Overshoot (\%) & Final-window error (V) \\",
        r"\hline",
    ]
    for r in summary_rows:
        latex_lines.append(
            f"{r['scenario']} & "
            f"{fmt_mean_std(r['rmse_mean'], r['rmse_std'])} & "
            f"{fmt_mean_std(r['overshoot_mean'], r['overshoot_std'])} & "
            f"{fmt_mean_std(r['final_window_error_mean'], r['final_window_error_std'])} \\\\"
        )
    latex_lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex_path = out_dir / "table8_td3_seed_statistics_latex.txt"
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"[OK] Raw seed results: {raw_path}")
    print(f"[OK] Seed summary CSV: {summary_path}")
    print(f"[OK] LaTeX table: {latex_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="run_boost_td3_experiment_revised.py", help="Main experiment script.")
    parser.add_argument("--recompute", default="recompute_tables_4_6_consistent.py", help="Metric recomputation script.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds to run.")
    parser.add_argument("--mode", default="full", help="Training mode passed to source script.")
    parser.add_argument("--out", default="three_seed_results_v3", help="Output folder for all seed results.")
    parser.add_argument("--actor_name", default="td3_actor.pt", help="Actor file name inside each results_seed folder.")
    args = parser.parse_args()

    root = Path.cwd()
    source = root / args.source
    recompute = root / args.recompute
    out_dir = root / args.out
    logs_dir = out_dir / "logs"
    metrics_root = out_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        print(f"[ERROR] Source script not found: {source}")
        sys.exit(1)
    if not recompute.exists():
        print(f"[ERROR] Recompute script not found: {recompute}")
        sys.exit(1)

    module_name = source.stem
    all_results = []

    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"[SEED {seed}] Full training and recomputation")
        print("=" * 80)

        train_out = root / f"results_seed_{seed}"
        train_log = logs_dir / f"train_seed{seed}.log"
        train_cmd = [
            sys.executable,
            str(source),
            "--mode", args.mode,
            "--seed", str(seed),
            "--out", str(train_out),
        ]
        code = run_command(train_cmd, cwd=root, log_file=train_log)
        if code != 0:
            print(f"[ERROR] Training failed for seed {seed}. Check log: {train_log}")
            sys.exit(code)

        actor_path = train_out / args.actor_name
        if not actor_path.exists():
            print(f"[ERROR] Expected actor not found: {actor_path}")
            print("Check the training log to see where the actor was saved.")
            sys.exit(1)

        metrics_dir = metrics_root / f"seed{seed}"
        metrics_log = logs_dir / f"recompute_seed{seed}.log"
        recompute_cmd = [
            sys.executable,
            str(recompute),
            "--module", module_name,
            "--actor", str(actor_path),
            "--out", str(metrics_dir),
        ]
        code = run_command(recompute_cmd, cwd=root, log_file=metrics_log)
        if code != 0:
            print(f"[ERROR] Recompute failed for seed {seed}. Check log: {metrics_log}")
            sys.exit(code)

        seed_results = extract_td3_metrics(metrics_dir, seed)
        all_results.extend(seed_results)
        print(f"[SEED {seed}] Done")

    write_summary(all_results, out_dir)
    print("\nAll seed experiments completed successfully.")
    print(f"Results folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
