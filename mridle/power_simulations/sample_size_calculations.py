import itertools
from power_simulations.sample_size_utilities import PowerSimulations
import os

precision = 0.513
recall = 0.429
avg_appts_per_week = 166  # taken from aggregation of df_features_original data for the year 2017 (in notebook #52)


effect_sizes = [0.1, 0.15, 0.2]
sample_sizes = [avg_appts_per_week * (i+1) for i in range(10)]
effect_sample_sizes = list(itertools.product(effect_sizes, sample_sizes))

effect_sizes = [0.1, 0.15, 0.2]
sample_sizes = [166 * (i+1) for i in range(10)]
exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=10,
                       num_runs_for_power_calc=10, original_test_set_length=4000, significance_level=0.05,
                       base_recall=recall, base_precision=precision, num_cpus=1, random_seed=0)


exp.run()
exp.save(parent_directory=os.getcwd())
