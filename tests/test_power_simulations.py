import unittest
from mridle.power_simulations.sample_size_utilities import PowerSimulations, calculate_f1_diff, permute_and_split
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from pandas.testing import assert_frame_equal

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
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)

        pooled = pd.concat([df, df_new])

        df_split, df_new_split = permute_and_split(pooled, split_point=exp.original_test_set_length)

        lengths = [len(df), len(df_new)]
        split_lengths = [len(df_split), len(df_new_split)]
        print(lengths, split_lengths)
        self.assertEqual(lengths, split_lengths)

    def test_pooling_splitting_distinct(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)
        pre_split_df = pd.concat([df, df_new])
        pre_split_f1 = f1_score(pre_split_df['true'], pre_split_df['pred'])

        pooled = pd.concat([df, df_new])

        df_split, df_new_split = permute_and_split(pooled, split_point=exp.original_test_set_length)
        post_split_df = pd.concat([df_split, df_new_split])
        post_split_f1 = f1_score(post_split_df['true'], post_split_df['pred'])

        test_condition = assert_frame_not_equal(df, df_split) & (pre_split_f1 == post_split_f1)

        self.assertTrue(test_condition)

    def test_f1_same_df(self):
        df = PowerSimulations.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)

        score = calculate_f1_diff(df, df)
        self.assertEqual(score, 0)

    def test_f1_diff_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)

        score = calculate_f1_diff(df, df_new)
        print(score)

        self.assertNotEqual(score, 0)

    def test_f1_diff_greater_than_0(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)

        score = calculate_f1_diff(df, df_new)
        print(score)
        self.assertGreater(score, 0)

    def test_generate_data_prec_orig_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        generated_prec = precision_score(df['true'], df['pred'])
        print(precision, generated_prec)
        self.assertAlmostEqual(precision, generated_prec, places=2)
        self.assertTrue(precision*0.97 <= generated_prec <= precision*1.03)

    def test_generate_data_rec_orig_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        df = exp.generate_actuals_preds(exp.base_precision, exp.base_recall, exp.original_test_set_length)
        generated_rec = recall_score(df['true'], df['pred'])
        print(recall*0.97, recall*1.03, generated_rec)
        self.assertTrue(recall*0.97 <= generated_rec <= recall*1.03)

    def test_generate_data_prec_new_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)
        generated_prec = precision_score(df_new['true'], df_new['pred'])

        print(prec_new, generated_prec)
        self.assertTrue(prec_new*0.97 <= generated_prec <= prec_new*1.03)

    def test_generate_data_rec_new_df(self):
        exp = PowerSimulations(sample_sizes=sample_sizes, effect_sizes=effect_sizes, num_trials_per_run=1000,
                               num_runs_for_power_calc=1000, original_test_set_length=4000, significance_level=0.05,
                               base_recall=recall, base_precision=precision, num_cpus=8, random_seed=0)

        prec_new = exp.base_precision * (1 - 0.1)
        recall_new = exp.base_recall * (1 - 0.1)
        sample_size_new = avg_appts_per_week * 5

        df_new = exp.generate_actuals_preds(prec_new, recall_new, sample_size_new)

        generated_rec = recall_score(df_new['true'], df_new['pred'])
        print(recall_new, generated_rec)
        self.assertTrue(recall_new*0.97 <= generated_rec <= recall_new*1.03)


if __name__ == '__main__':
    unittest.main()
