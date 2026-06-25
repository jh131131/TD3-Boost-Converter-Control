#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Six-seed TD3 experiment pipeline for the boost converter manuscript.

Purpose
-------
This script extends the original three-seed TD3 analysis to six random seeds
without introducing SAC or changing the main experimental design. It trains or
reuses TD3 actors for seeds 0--5, recomputes the three disturbance-scenario
metrics, aggregates the mean and standard deviation, and generates a revised
training convergence figure with mean curve and standard-deviation band.

Recommended usage from the project root:
    python run_td3_six_seed_pipeline.py \
        --source code/run_boost_td3_experiment_revised.py \
        --recompute code/recompute_tables_4_6_with_awpi.py \
        --seeds 0 1 2 3 4 5 \
        --mode full \
        --out six_seed_results

If results_seed_0/results_seed_1/results_seed_2 already exist in the project
root, the script will reuse them by default and only train missing seeds.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Use a non-interactive backend for servers/Overleaf-like environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def run_command(cmd: List[str], cwd: Path, log_file: Path, extra_pythonpath: Optional[Path] = None) -> int:
    print("[RUN]", " ".join(str(x) for x in cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_pythonpath is not None:
        old_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(extra_pythonpath) + (os.pathsep + old_path if old_path else "")
    with log_file.open("w", encoding="utf-8", errors="ignore") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
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
            name_norm = name.lower().replace(" ", "_")
            if name in TD3_NAMES or name_norm in TD3_NAMES:
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
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return mean, std


def fmt_mean_std(mean: Optional[float], std: Optional[float], digits: int = 2) -> str:
    if mean is None or std is None:
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_seed_summary(results: List[dict], out_dir: Path, n_seeds: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "td3_six_seed_raw_results.csv"
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

    summary_path = out_dir / "td3_six_seed_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Statistical performance of the proposed TD3 controller under {n_seeds} random seeds.}}",
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
    latex_lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    latex_path = out_dir / "table_td3_six_seed_statistics_latex.txt"
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"[OK] Raw seed results: {raw_path}")
    print(f"[OK] Seed summary CSV: {summary_path}")
    print(f"[OK] LaTeX table: {latex_path}")


def find_existing_train_dir(project_root: Path, seed: int, preferred_dir: Path, actor_name: str, rerun_existing: bool) -> Optional[Path]:
    if rerun_existing:
        return None
    candidates = [
        preferred_dir,
        project_root / f"results_seed_{seed}",
        project_root / "code" / f"results_seed_{seed}",
    ]
    for d in candidates:
        if (d / actor_name).exists() and (d / "training_log.csv").exists():
            return d
    return None


def read_training_log(path: Path) -> Dict[str, np.ndarray]:
    rows = read_csv_dicts(path)
    if not rows:
        raise RuntimeError(f"Empty training log: {path}")
    cols = rows[0].keys()
    required = ["episode", "average_reward", "voltage_rmse", "average_duty_variation"]
    for c in required:
        if c not in cols:
            raise RuntimeError(f"Column {c} not found in {path}")
    return {c: np.asarray([float(r[c]) for r in rows], dtype=float) for c in required}


def create_training_mean_std_figure(train_dirs: Dict[int, Path], out_dir: Path) -> None:
    logs = []
    for seed, d in sorted(train_dirs.items()):
        log_path = d / "training_log.csv"
        if not log_path.exists():
            print(f"[WARN] Training log missing for seed {seed}: {log_path}")
            continue
        logs.append(read_training_log(log_path))
    if len(logs) < 2:
        print("[WARN] Not enough training logs to create mean/std Fig. 8.")
        return

    min_len = min(len(x["episode"]) for x in logs)
    episodes = logs[0]["episode"][:min_len]

    def stack_metric(name: str) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.vstack([x[name][:min_len] for x in logs])
        return np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)

    reward_mean, reward_std = stack_metric("average_reward")
    rmse_mean, rmse_std = stack_metric("voltage_rmse")
    duty_mean, duty_std = stack_metric("average_duty_variation")

    fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
    ax1.plot(episodes, reward_mean, linewidth=2.0, label="Average reward (mean)")
    ax1.fill_between(episodes, reward_mean - reward_std, reward_mean + reward_std, alpha=0.18, label="Reward $\\pm$ 1 std")
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Average reward")
    ax1.grid(True, linestyle=":", linewidth=0.7)

    ax2 = ax1.twinx()
    ax2.plot(episodes, rmse_mean, linewidth=1.6, linestyle="--", label="Voltage RMSE (mean)")
    ax2.fill_between(episodes, rmse_mean - rmse_std, rmse_mean + rmse_std, alpha=0.10, label="RMSE $\\pm$ 1 std")
    ax2.plot(episodes, duty_mean, linewidth=1.6, linestyle="-.", label="Duty variation (mean)")
    ax2.set_ylabel("RMSE / duty variation")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig8_training_convergence_six_seed.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig8_training_convergence_six_seed.jpeg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Six-seed Fig. 8 saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".", help="Project root containing the code/data folders.")
    parser.add_argument("--source", default="code/run_boost_td3_experiment_revised.py", help="Main experiment script.")
    parser.add_argument("--recompute", default="code/recompute_tables_4_6_with_awpi.py", help="Metric recomputation script.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5], help="Random seeds to run or reuse.")
    parser.add_argument("--mode", default="full", choices=["quick", "full"], help="Training mode passed to the source script.")
    parser.add_argument("--out", default="six_seed_results", help="Output folder for aggregated six-seed results.")
    parser.add_argument("--actor_name", default="td3_actor.pt", help="Actor file name inside each results_seed folder.")
    parser.add_argument("--rerun_existing", action="store_true", help="Retrain even if a seed result already exists.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    source = (project_root / args.source).resolve()
    recompute = (project_root / args.recompute).resolve()
    out_dir = (project_root / args.out).resolve()
    training_root = out_dir / "training"
    metrics_root = out_dir / "metrics"
    logs_dir = out_dir / "logs"
    figures_dir = out_dir / "figures"
    summary_dir = out_dir / "summary"

    if not source.exists():
        raise FileNotFoundError(f"Source script not found: {source}")
    if not recompute.exists():
        raise FileNotFoundError(f"Recompute script not found: {recompute}")

    source_dir = source.parent
    module_name = source.stem
    train_dirs: Dict[int, Path] = {}
    all_results: List[dict] = []

    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"[SEED {seed}] Training/reuse and metric recomputation")
        print("=" * 80)

        preferred_train_dir = training_root / f"results_seed_{seed}"
        existing_train_dir = find_existing_train_dir(project_root, seed, preferred_train_dir, args.actor_name, args.rerun_existing)
        if existing_train_dir is not None:
            train_out = existing_train_dir
            print(f"[SKIP TRAINING] Reusing existing seed result: {train_out}")
        else:
            train_out = preferred_train_dir
            train_cmd = [sys.executable, str(source), "--mode", args.mode, "--seed", str(seed), "--out", str(train_out)]
            code = run_command(train_cmd, cwd=source_dir, log_file=logs_dir / f"train_seed{seed}.log", extra_pythonpath=source_dir)
            if code != 0:
                raise RuntimeError(f"Training failed for seed {seed}. Check log: {logs_dir / f'train_seed{seed}.log'}")

        actor_path = train_out / args.actor_name
        if not actor_path.exists():
            raise FileNotFoundError(f"Expected actor not found for seed {seed}: {actor_path}")
        train_dirs[seed] = train_out

        metrics_dir = metrics_root / f"seed{seed}"
        expected_metrics = [metrics_dir / f for f in SCENARIO_TABLES.values()]
        if not args.rerun_existing and all(p.exists() for p in expected_metrics):
            print(f"[SKIP RECOMPUTE] Reusing metrics: {metrics_dir}")
        else:
            recompute_cmd = [
                sys.executable,
                str(recompute),
                "--module", module_name,
                "--actor", str(actor_path),
                "--out", str(metrics_dir),
            ]
            code = run_command(recompute_cmd, cwd=source_dir, log_file=logs_dir / f"recompute_seed{seed}.log", extra_pythonpath=source_dir)
            if code != 0:
                raise RuntimeError(f"Metric recomputation failed for seed {seed}. Check log: {logs_dir / f'recompute_seed{seed}.log'}")

        seed_results = extract_td3_metrics(metrics_dir, seed)
        all_results.extend(seed_results)

    write_seed_summary(all_results, summary_dir, n_seeds=len(args.seeds))
    create_training_mean_std_figure(train_dirs, figures_dir)

    print("\nAll requested seed experiments completed successfully.")
    print(f"Aggregated results folder: {out_dir}")


if __name__ == "__main__":
    main()
