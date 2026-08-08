# TD3-Based State-Dependent Anti-Disturbance Control for a DC-DC Boost Converter

This repository contains the revised simulation code, trained models, processed data, statistical results, and regenerated figures supporting the manuscript **“TD3-Based State-Dependent Anti-Disturbance Control for a DC-DC Boost Converter.”**

## Key revision points

- The boost-converter averaged model is integrated with fourth-order Runge–Kutta (RK4).
- Controller update period: **500 μs**.
- Default RK4 internal step: **50 μs**; numerical-sensitivity checks also use **25 μs** and **10 μs**.
- TD3 and DDPG are **independently trained**; the previous placeholder `DDPG_like` comparison is not used in the revised manuscript.
- TD3 and DDPG use matched state, reward, action bounds, actor/critic widths, learning rates, replay settings, training distributions, and six matched random seeds **0–5**.
- Conventional PI/PI-AW and SMC gains are selected on validation disturbance schedules that differ from the final test schedules.
- The revised results do **not** claim universal TD3 superiority. Across six matched seeds, TD3 and DDPG have broadly comparable performance and the paired exact permutation tests do not show significance at α = 0.05.

## Environment

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

The code uses NumPy, Matplotlib, and PyTorch. CPU execution is sufficient for the reported small neural networks.

## Repository structure

- `code/boost_td3_ddpg_revised_experiment.py`: main end-to-end revised experiment; includes the RK4 environment, TD3, true DDPG, PI/PI-AW, SMC, baseline validation tuning, six-seed evaluation, exact paired permutation tests, main tables, and Figs. 8–13 except the observation ablation.
- `code/train_one_agent.py`: train a single TD3 or DDPG seed.
- `code/tune_validation.py`: reproduce validation-based PI/PI-AW and SMC gain selection.
- `code/train_masked_td3.py`: train observation-signal ablations (`no_vin`, `no_load`, `no_both`).
- `code/eval_masked_ablation.py`: evaluate the observation-signal ablations.
- `code/run_full_revision_pipeline.py`: convenience script for the revised main experiment plus all three observation ablations.
- `data/tables/`: manuscript Tables 4–6 from the representative trajectory evaluation.
- `data/trajectories/`: representative seed-0 time-series data (output voltage, duty cycle, inductor current, bound-hit flag) for PI, PI-AW, SMC, DDPG, and TD3 under all three final scenarios.
- `data/six_seed/`: matched TD3/DDPG six-seed raw results, summary statistics, and exact paired permutation-test p-values.
- `data/ablation/`: observation-signal ablation results.
- `data/numerical_sensitivity/`: RK4 integration-step sensitivity results.
- `data/config/`: final experiment configuration.
- `models/training/`: six TD3 and six DDPG trained actors with training logs.
- `models/ablation/`: trained observation-ablation actors and logs.
- `figures/`: regenerated manuscript Figs. 7–13. Existing Figs. 1–6 are unchanged from the previous repository version.

## Reproduce the revised main experiment

From the repository root:

```bash
python code/boost_td3_ddpg_revised_experiment.py \
  --out reproduced_results \
  --episodes 20 \
  --steps 500 \
  --seeds 0 1 2 3 4 5
```

This trains independent TD3 and DDPG agents, evaluates all six matched seeds, tunes the conventional controllers on separate validation schedules, generates the representative-scenario tables and figures, and evaluates RK4 integration-step sensitivity.

To reproduce the observation ablations as well:

```bash
python code/run_full_revision_pipeline.py --out reproduced_results
```

## Final experimental configuration

- Nominal input voltage: 100 V
- Reference output voltage: 200 V
- Inductance: 1 mH
- Output capacitance: 470 μF
- Nominal load resistance: 50 Ω
- Switching frequency used for the converter specification: 20 kHz
- Controller update period: 500 μs
- RK4 internal integration step: 50 μs
- Duty-cycle bounds: [0.05, 0.90]
- Actor/critic hidden layers: 128–128, ReLU
- Actor learning rate: 1e-4
- Critic learning rate: 1e-3
- Discount factor γ: 0.99
- Soft-update coefficient τ: 0.005
- Replay buffer: 250,000 transitions
- Mini-batch size: 128
- Exploration-noise standard deviation: 0.03
- TD3 target-policy noise: 0.02
- TD3 target-noise clip: 0.05
- TD3 policy delay: 2
- Training budget: 20 episodes × 500 controller steps
- Random seeds: 0, 1, 2, 3, 4, 5
- PI / PI-AW: Kp = 5e-5, Ki = 0.03
- SMC: k = 0.11, λ = 0.50
- Representative response seed: 0, selected by distance to the component-wise median TD3 RMSE vector rather than by best performance.

## Important scope statement

The reported results are simulation-based and use an averaged CCM boost-converter model. The repository does not establish Lyapunov stability, a certified current-safety filter, hardware/HIL performance, sensor-noise robustness, or measured embedded-controller inference latency. These limitations are stated explicitly in the revised manuscript.
