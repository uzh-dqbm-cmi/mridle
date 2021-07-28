import unittest
from mridle.power_simulations.sample_size_utilities import PowerSimulations, calculate_f1_diff, permute_and_split
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from pandas.testing import assert_frame_equal
import numpy as np


precision = 0.513
recall = 0.429

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
        # split out into two tests, one for the splitting (Create dummy df and do it that way),
        # and another for testing that same dfs aren't returned
        df = PowerSimulations.generate_actuals_preds(precision, recall, 4000)

        df_split_1, df_split_2 = permute_and_split(df, split_point=3000)

        recombined = pd.concat([df_split_1, df_split_2])
        recombined.sort_index(inplace=True)
        assert_frame_equal(df, recombined)

    def test_permute_split_not_same_df_returned(self):
        df_1 = PowerSimulations.generate_actuals_preds(precision, recall, 4000)
        df_2 = PowerSimulations.generate_actuals_preds(precision, recall, 1000)
        pooled = pd.concat([df_1, df_2])
        df_split_1, df_split_2 = permute_and_split(pooled, split_point=4000)

        assert_frame_not_equal(df_1, df_split_1)

    def test_f1_same_df(self):
        df = PowerSimulations.generate_actuals_preds(precision, recall, 4000)

        score = calculate_f1_diff(df, df)
        self.assertEqual(score, 0)

    def test_f1_diff_df(self):
        df = PowerSimulations.generate_actuals_preds(precision, recall, 4000)

        prec_new = precision * (1 - 0.1)
        recall_new = recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds(prec_new, recall_new, sample_size_new)

        score = calculate_f1_diff(df, df_new)

        self.assertNotEqual(score, 0)

    def test_f1_diff_greater_than_0(self):
        df = PowerSimulations.generate_actuals_preds(precision, recall, 4000)

        prec_new = precision * (1 - 0.1)
        recall_new = recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = PowerSimulations.generate_actuals_preds(prec_new, recall_new, sample_size_new)

        score = calculate_f1_diff(df, df_new)

        self.assertGreater(score, 0)

    def test_generate_data_precision_orig_test_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        generated_prec = precision_score(df['true'], df['pred'])

        self.assertTrue(precision*0.97 <= generated_prec <= precision*1.03)

    def test_generate_data_recall_orig_test_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        generated_rec = recall_score(df['true'], df['pred'])

        self.assertTrue(recall*0.97 <= generated_rec <= recall*1.03)

    def test_generate_data_precision_new_test_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)
        generated_prec = precision_score(df_new['true'], df_new['pred'])

        self.assertTrue(prec_new*0.97 <= generated_prec <= prec_new*1.03)

    def test_generate_data_df_proportions(self):
        class_0_proportion = 0.86
        df = PowerSimulations.generate_actuals_preds(precision, recall, 166, p=[class_0_proportion,
                                                                                1-class_0_proportion])
        class_0 = np.sum(df['true'] == 0) / len(df)

        self.assertTrue(class_0_proportion*0.97 <= class_0 <= class_0_proportion*1.03)


if __name__ == '__main__':
    unittest.main()
