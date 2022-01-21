import os
from mridle.utilities.power_simulations.sample_size_utilities import PowerSimulations

performance = 0.42
performance_type = 'log_loss'
avg_appts_per_week = 166  # taken from aggregation of df_features_original data for the year 2017 (in notebook #52)


effect_sizes = [0.01, 0.025, 0.05, 0.1]
sample_sizes = [avg_appts_per_week * (i+1) for i in range(10)]


exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=100,
                       num_runs_for_power_calc=100, original_test_set_length=6536, significance_level=0.05,
                       base_performance=performance, performance_type=performance_type, num_cpus=8, random_seed=0)

exp.run()
exp.save(parent_directory=os.getcwd())
