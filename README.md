# TD3-Based Anti-Disturbance Control of a DC-DC Boost Converter

This repository contains the simulation code, processed data, and generated results supporting the manuscript titled **Research on Anti-disturbance Adaptive Control Strategy of Power Electronic Converters Based on Deep Reinforcement Learning**.

## Repository structure

- `code/`: Python scripts used for TD3 training, baseline comparison, metric recomputation, and additional response figures.
- `data/tables/`: CSV files corresponding to the main training and disturbance-comparison tables in the manuscript.
- `data/trajectories/`: Output-voltage, duty-cycle, and related trajectory data under load disturbance, input-voltage fluctuation, and parameter variation.
- `data/six_seed/`: Raw, summarized, and archived results for the six random-seed robustness analysis.
- `data/three_seed/`: Earlier three-seed robustness files retained for traceability.
- `figures/`: Generated manuscript figures in JPEG format.
- `models/`: Trained TD3 actor network used in the reported simulation tests.

## Main files used in the manuscript

- Table 3: `data/tables/table3_training_performance.csv`
- Table 4: `data/tables/table4_load_disturbance.csv`
- Table 5: `data/tables/table5_input_fluctuation.csv`
- Table 6: `data/tables/table6_parameter_variation.csv`
- Six-seed statistics: `data/six_seed/td3_six_seed_summary.csv`
- Six-seed raw metrics: `data/six_seed/td3_six_seed_raw_results.csv`
- Table 7 LaTeX text: `data/six_seed/table_td3_six_seed_statistics_latex.txt`
- Fig. 1: `figures/fig1_boost_topology.jpeg`
- Fig. 2: `figures/fig2_switching_modes_final.jpeg`
- Fig. 3: `figures/fig3_disturbance_modeling.jpeg`
- Fig. 4: `figures/fig4_overall_framework.jpeg`
- Fig. 5: `figures/fig5_mdp_formulation.jpeg`
- Fig. 6: `figures/fig6_td3_architecture.jpeg`
- Fig. 7: `figures/fig7_disturbance_scenarios.jpeg`
- Fig. 8: `figures/fig8_training_convergence_six_seed.jpeg`
- Fig. 9: `figures/fig9_dynamic_response_comparison.jpeg`
- Fig. 10: `figures/fig10_output_voltage_response_details.jpeg`
- Fig. 11: `figures/fig11_duty_cycle_trajectories.jpeg`
- Fig. 12: `figures/fig12_absolute_tracking_error_trajectories_zoomed.jpeg`

## Environment

The scripts require Python 3.10 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility note

The manuscript reports a simulation-based verification using an averaged continuous-conduction-mode boost-converter model. The uploaded files include processed results and trajectory data used to generate the tables and figures in the manuscript. The updated robustness analysis uses six independent random seeds, and the corresponding Fig. 8 and Table 7 files are included in `figures/` and `data/six_seed/`.
