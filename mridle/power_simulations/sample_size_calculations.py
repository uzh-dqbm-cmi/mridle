import os
from mridle.power_simulations.sample_size_utilities import PowerSimulations

performance = 0.65
avg_appts_per_week = 166  # taken from aggregation of df_features_original data for the year 2017 (in notebook #52)


effect_sizes = [0.1, 0.15, 0.2]
sample_sizes = [avg_appts_per_week * (i+1) for i in range(10)]


exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=10,
                       num_runs_for_power_calc=10, original_test_set_length=4000, significance_level=0.05,
                       base_performance=performance, performance_type='f1_macro', num_cpus=8, random_seed=0)

exp.run()
exp.save(parent_directory=os.getcwd())
