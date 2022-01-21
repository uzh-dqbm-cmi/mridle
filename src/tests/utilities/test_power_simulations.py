import unittest
from mridle.utilities.power_simulations.sample_size_utilities import PowerSimulations, calculate_f1_diff, \
    permute_and_split, calculate_log_loss_diff
import pandas as pd
from sklearn.metrics import f1_score, precision_score, log_loss
from pandas.testing import assert_frame_equal
import numpy as np

p = 0.14
avg_appts_per_week = 166

effect_sizes = [0.1, 0.15, 0.2]
sample_sizes = [avg_appts_per_week * (i + 1) for i in range(10)]


def assert_frame_not_equal(*args, **kwargs):
    try:
        assert_frame_equal(*args, **kwargs)
    except AssertionError:
        # frames are not equal
        return True
    else:
        # frames are equal
        raise AssertionError


class TestPowerSimulations(unittest.TestCase):
    def test_pooling_splitting_lengths(self):
        df = pd.DataFrame(np.random.randint(0, 2, size=(200, 1)), columns=['test'])
        df_new = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)), columns=['test'])

        pooled = pd.concat([df, df_new])

        df_split, df_new_split = permute_and_split(pooled, split_point=len(df))

        lengths = [len(df), len(df_new)]
        split_lengths = [len(df_split), len(df_new_split)]

        self.assertEqual(lengths, split_lengths)

    def test_permute_split_no_resampling_of_rows(self):
        performance = 0.513

        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000, p=p)

        df_split_1, df_split_2 = permute_and_split(df, split_point=3000)

        recombined = pd.concat([df_split_1, df_split_2])
        recombined.sort_index(inplace=True)
        assert_frame_equal(df, recombined)

    def test_permute_split_not_same_df_returned(self):
        performance = 0.513
        df_1 = PowerSimulations.generate_actuals_preds_precision(performance, 4000, p=p)
        df_2 = PowerSimulations.generate_actuals_preds_precision(performance, 1000, p=p)
        pooled = pd.concat([df_1, df_2])
        df_split_1, df_split_2 = permute_and_split(pooled, split_point=4000)

        assert_frame_not_equal(df_1, df_split_1)

    def test_f1_same_df(self):
        performance = 0.513

        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000, p=p)

        score = calculate_f1_diff(df, df)
        self.assertEqual(score, 0)

    def test_f1_diff_df(self):
        performance = 0.513

        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000, p=p)

        performance_new = performance * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds_precision(performance_new, sample_size_new, p=p)

        score = calculate_f1_diff(df, df_new)

        self.assertNotEqual(score, 0)

    def test_f1_diff_greater_than_0(self):
        performance = 0.513
        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000, p=p)

        performance_new = performance * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds_precision(performance_new, sample_size_new, p=p)

        score = calculate_f1_diff(df, df_new)

        self.assertGreater(score, 0)

    def test_log_loss_same_df(self):
        performance = 0.42

        df = PowerSimulations.generate_actuals_preds_log_loss(performance, 4000, p=p)

        score = calculate_log_loss_diff(df, df)
        self.assertEqual(score, 0)

    def test_log_loss_diff_df(self):
        performance = 0.42

        df = PowerSimulations.generate_actuals_preds_log_loss(performance, 4000, p=p)

        performance_new = performance * (1 + 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds_log_loss(performance_new, sample_size_new, p=p)

        score = calculate_log_loss_diff(df, df_new)

        self.assertNotEqual(score, 0)

    def test_log_loss_diff_greater_than_0(self):
        performance = 0.42
        df = PowerSimulations.generate_actuals_preds_log_loss(performance, 4000, p=p)

        performance_new = performance * (1 + 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds_log_loss(performance_new, sample_size_new, p=p)

        score = calculate_log_loss_diff(df, df_new)

        self.assertGreater(score, 0)

    def test_generate_data_precision(self):
        performance = 0.513
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=performance, performance_type='precision', num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds_precision(exp.base_performance, exp.original_test_set_length, p=p)
        generated_prec = precision_score(df['true'], df['pred'])

        self.assertTrue(performance*0.97 <= generated_prec <= performance*1.03)

    def test_generate_data_f1_macro(self):
        performance = 0.65
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=performance, performance_type='f1_macro', num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds_f1_macro(exp.base_performance, exp.original_test_set_length, p=p)
        generated_f1_macro = f1_score(df['true'], df['pred'], average='macro')

        self.assertTrue(performance*0.97 <= generated_f1_macro <= performance*1.03)

    def test_generate_data_log_loss(self):
        truth_list = []
        for performance in np.arange(0.3, 0.6, 0.01):
            for k in range(100):
                exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                                       num_runs_for_power_calc=1000, original_test_set_length=4000,
                                       significance_level=0.05, base_performance=performance,
                                       performance_type='log_loss', num_cpus=8, random_seed=0)

                df = exp.generate_actuals_preds_log_loss(exp.base_performance, exp.original_test_set_length, p=p)
                generated_log_loss = log_loss(df['true'], df['pred'])
                within_range = performance*0.99 <= generated_log_loss <= performance*1.01
                truth_list.append(within_range)
        self.assertTrue(all(truth_list))

    def test_generate_data_df_proportions_precision(self):
        class_1_proportion = 0.2
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=0.6, performance_type='precision', num_cpus=8, random_seed=0,
                               p=class_1_proportion)

        df = exp.generate_actuals_preds_precision(exp.base_performance, exp.original_test_set_length, p=exp.p)
        class_1 = np.sum(df['true']) / len(df)
        self.assertTrue(exp.p*0.90 <= class_1 <= exp.p*1.1)

    def test_generate_data_df_proportions_f1_macro(self):
        class_1_proportion = 0.2
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=0.6, performance_type='f1_macro', num_cpus=8, random_seed=0,
                               p=class_1_proportion)

        df = exp.generate_actuals_preds_f1_macro(exp.base_performance, exp.original_test_set_length, p=exp.p)
        class_1 = np.sum(df['true']) / len(df)

        self.assertTrue(exp.p*0.92 <= class_1 <= exp.p*1.08)

    def test_generate_data_df_proportions_log_loss(self):
        class_1_proportion = 0.17
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=0.6, performance_type='log_loss', num_cpus=8, random_seed=0,
                               p=class_1_proportion)

        df = exp.generate_actuals_preds_log_loss(exp.base_performance, exp.original_test_set_length, p=exp.p)
        class_1 = np.sum(df['true']) / len(df)

        self.assertTrue(exp.p*0.99 <= class_1 <= exp.p*1.01)


if __name__ == '__main__':
    unittest.main()
