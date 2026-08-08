# Reproducibility notes for the revised manuscript

The revised experimental package supersedes the earlier preliminary scripts and processed results used before peer review.

## What changed

1. Forward-Euler discretization was replaced by RK4 integration with a 500-μs controller period and 50-μs internal step.
2. The previous placeholder `DDPG_like` branch was removed from the evidence base and replaced by an independently trained DDPG implementation.
3. TD3 and DDPG were trained under the same six random seeds (0–5), and uncertainty plus exact paired permutation tests are reported.
4. PI/PI-AW and SMC parameters were selected using validation disturbances that differ from the final test schedules.
5. The old 320-V saturation-limited 60% overshoot comparisons were removed. The revised tests do not hit the voltage numerical bound.
6. Recovery time is measured from the final disturbance transition, not from an unspecified evaluation point.
7. Separate observation ablations quantify the effect of the input-voltage and load-observation channels.
8. RK4 internal-step sensitivity is evaluated at 50, 25, and 10 μs.

## Files corresponding to the revised manuscript

- Main disturbance tables: `data/tables/`
- Representative response trajectories: `data/trajectories/`
- Six-seed TD3/DDPG statistics: `data/six_seed/`
- Observation ablation: `data/ablation/observation_ablation.csv`
- Numerical sensitivity: `data/numerical_sensitivity/integration_step_sensitivity.csv`
- Regenerated figures: `figures/fig7_*.jpeg` through `figures/fig13_*.jpeg`
- Trained actors and logs: `models/`

The Git history may retain earlier preliminary files, but the current `main` branch should use the revised files listed above for manuscript reproduction.
