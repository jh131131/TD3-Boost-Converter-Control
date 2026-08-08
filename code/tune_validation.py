from boost_td3_ddpg_revised_experiment import tune_baselines

if __name__ == "__main__":
    pi_gains, smc_gains = tune_baselines()
    print(f"Validation-selected PI/PI-AW gains: Kp={pi_gains[0]}, Ki={pi_gains[1]}")
    print(f"Validation-selected SMC gains: k={smc_gains[0]}, lambda={smc_gains[1]}")
