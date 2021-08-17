import unittest
from mridle.power_simulations.sample_size_utilities import PowerSimulations, calculate_f1_diff, permute_and_split
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from pandas.testing import assert_frame_equal
import numpy as np


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

        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000)

        df_split_1, df_split_2 = permute_and_split(df, split_point=3000)

        recombined = pd.concat([df_split_1, df_split_2])
        recombined.sort_index(inplace=True)
        assert_frame_equal(df, recombined)

    def test_permute_split_not_same_df_returned(self):
        performance = 0.513
        df_1 = PowerSimulations.generate_actuals_preds_precision(performance, 4000)
        df_2 = PowerSimulations.generate_actuals_preds_precision(performance, 1000)
        pooled = pd.concat([df_1, df_2])
        df_split_1, df_split_2 = permute_and_split(pooled, split_point=4000)

        assert_frame_not_equal(df_1, df_split_1)

    def test_f1_same_df(self):
        performance = 0.513

        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000)

        score = calculate_f1_diff(df, df)
        self.assertEqual(score, 0)

    def test_f1_diff_df(self):
        performance = 0.513

        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000)

        performance_new = performance * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds_precision(performance_new, sample_size_new)

        score = calculate_f1_diff(df, df_new)

        self.assertNotEqual(score, 0)

    def test_f1_diff_greater_than_0(self):
        performance = 0.513
        df = PowerSimulations.generate_actuals_preds_precision(performance, 4000)

        performance_new = performance * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds_precision(performance_new, sample_size_new)

        score = calculate_f1_diff(df, df_new)

        self.assertGreater(score, 0)

    def test_generate_data_precision(self):
        performance = 0.513
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=performance, performance_type='precision', num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds_precision(exp.base_performance, exp.original_test_set_length)
        generated_prec = precision_score(df['true'], df['pred'])

        self.assertTrue(performance*0.97 <= generated_prec <= performance*1.03)

    def test_generate_data_f1_macro(self):
        performance = 0.65
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_performance=performance, performance_type='precision', num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds_f1_macro(exp.base_performance, exp.original_test_set_length)
        generated_f1_macro = f1_score(df['true'], df['pred'], average='macro')

        self.assertTrue(performance*0.97 <= generated_f1_macro <= performance*1.03)

    def test_generate_data_df_proportions_precision(self):
        class_0_proportion = 0.86
        df = PowerSimulations.generate_actuals_preds_precision(0.513, 166, p=[class_0_proportion,
                                                                              1-class_0_proportion])
        class_0 = np.sum(df['true'] == 0) / len(df)

        self.assertTrue(class_0_proportion*0.92 <= class_0 <= class_0_proportion*1.08)

    def test_generate_data_df_proportions_f1_macro(self):
        class_0_proportion = 0.86
        df = PowerSimulations.generate_actuals_preds_f1_macro(0.6, 166, p=[class_0_proportion,
                                                                           1-class_0_proportion])
        class_0 = np.sum(df['true'] == 0) / len(df)

        self.assertTrue(class_0_proportion*0.95 <= class_0 <= class_0_proportion*1.05)


if __name__ == '__main__':
    unittest.main()
