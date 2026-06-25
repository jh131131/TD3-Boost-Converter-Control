# Six-seed update for TD3 robustness analysis

This repository has been updated to match the revised manuscript version in which the TD3 training robustness analysis uses six independent random seeds.

## Updated manuscript items

- **Fig. 8** has been replaced by the six-seed training convergence figure with mean curves and ±1 standard deviation shading.
- **Table 7** is supported by the six-seed summary statistics in `data/six_seed/td3_six_seed_summary.csv`.
- The raw six-seed scenario metrics are provided in `data/six_seed/td3_six_seed_raw_results.csv`.
- The LaTeX table text used for the manuscript is provided in `data/six_seed/table_td3_six_seed_statistics_latex.txt`.
- The complete six-seed result archive is provided as `data/six_seed/six_seed_results_archive.zip`.

## Six-seed statistical results reported in the manuscript

| Scenario | RMSE (V) | Overshoot (%) | Final-window error (V) |
|---|---:|---:|---:|
| Load disturbance | 8.82 ± 1.01 | 3.94 ± 0.56 | 6.91 ± 1.57 |
| Input voltage fluctuation | 4.96 ± 0.57 | 3.91 ± 0.52 | 1.76 ± 0.44 |
| Parameter variation | 1.49 ± 1.07 | 0.69 ± 0.60 | 1.49 ± 1.07 |

## Reproduction script

The script `code/run_td3_six_seed_pipeline.py` documents the six-seed training and metric-generation workflow used for the revised results.
