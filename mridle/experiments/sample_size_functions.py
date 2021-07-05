import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from multiprocessing import Pool
import itertools
from typing import List


class PowerSimulations:
    """
    Class for running power simulations for calculating the sample size, effect size, and power for the
    'silent live test'. A list of sample sizes and a list of effect sizes are declared when creating this
    object. These two lists are then used to calculate the power of our permutation test for each combination
    of elements in these two lists.
    """

    def __init__(self, sample_sizes, effect_sizes, num_permutation_runs, num_power_runs,
                 original_test_set_length, significance_level, base_precision, base_recall,
                 num_cpus):
        """
        Create a PowerSimulations objects.

        Args:
            sample_sizes: List of sample sizes to calculate the power for
            effect_sizes: List of effect sizes to calculate the power for
            num_permutation_runs: number of simulations to run for the permutation test to obtain the distribution of
            the null hypothesis (1000+ recommended if computer speed allows)
            num_power_runs: number of times to run the permutation experiment to obtain an estimate for the power of
            the test
            original_test_set_length: Number of samples to create for the 'original' test set
            significance_level: Significance level to use for the permutation test
            base_precision: Precision of the model on the original test set
            base_recall: Recall of the model on the original test set
            num_cpus: Number of cpus to use for the experiment executions
        """
        self.sample_sizes = sample_sizes
        self.effect_sizes = effect_sizes
        self.num_permutation_runs = num_permutation_runs
        self.num_power_runs = num_power_runs
        self.original_test_set_length = original_test_set_length
        self.significance_level = np.float64(significance_level)
        self.base_precision = base_precision
        self.base_recall = base_recall
        self.results = None
        self.num_cpus = num_cpus

    def run(self):
        """
        Run the permutation tests and save the results as an attribute of the object.
        """
        effect_sample_sizes = list(itertools.product(self.effect_sizes, self.sample_sizes))
        with Pool(self.num_cpus) as p:
            results = p.map(self.__run_helper, effect_sample_sizes)

        power = [np.sum(res < self.significance_level) / len(res) for res in results]
        self.results = pd.DataFrame(effect_sample_sizes, power)

    def __run_helper(self, effect_sample_sizes):
        """
        Helper function for running the permutation experiments, helping with parallelisation

        Args:
            effect_sample_sizes: list of tuples containing each combination of the declared effect and sample sizes

        Returns:
            List of alpha values for each test in the power analysis trials

        """
        effect_size, sample_size = effect_sample_sizes

        precision_new = self.base_precision * (1 - effect_size)
        recall_new = self.base_recall * (1 - effect_size)

        np.random.seed(0)
        alphas = [self.permutation_trials(precision_new, recall_new, sample_size) for i in range(self.num_power_runs)]

        return alphas

    def permutation_trials(self, prec_new, rec_new, sample_size_new):
        """
        Execute n=self.num_permutation_runs runs of the individual permuted trial in the function one_trial.
        An alpha value for this trial is returned, and we will then obtain n=self.num_power_runs values
        for this alpha value by running this function over and over again. Using this set of alpha values, we
        can then calculated the power of our test.

        Args:
            prec_new: Precision that the new test set and predictions should be generated with
            rec_new: Recall that the new test set and predictions should be generated with
            sample_size_new: Sample size of new test set to be generated

        Returns:
            Alpha value for one group of permutation tests.

        """
        df = self.generate_actuals_preds(self.base_precision, self.base_recall, self.original_test_set_length)
        df_new = self.generate_actuals_preds(prec_new, rec_new, sample_size_new)

        pooled = pd.concat([df, df_new])
        orig_diff = f1_score(df['true'], df['pred'], average='macro') - f1_score(df_new['true'], df_new['pred'],
                                                                                 average='macro')

        differences = [self.one_trial(pooled) for i in range(self.num_permutation_runs)]
        individual_alpha = np.sum(differences > orig_diff) / len(differences)

        return individual_alpha

    def one_trial(self, pooled_data):
        """
        Take in pooled data, create one set of permuted datasets and return the test statistic for this one trial
        Args:
            pooled_data: Both the 'original' test set and the new test set combined into one dataframe, which
            we then use to split into two random groups of the same size as each of the two individual datasets

        Returns:
            Test statistic for one run of the permutation trials

        """
        permuted = pooled_data.sample(frac=1)
        new_df = permuted[:self.original_test_set_length]
        new_df_new = permuted[self.original_test_set_length:]

        return f1_score(new_df['true'], new_df['pred'], average='macro') - f1_score(new_df_new['true'],
                                                                                    new_df_new['pred'], average='macro')

    @staticmethod
    def generate_actuals_preds(prec: float, rec: float, n: int, p: List[float] = None) -> pd.DataFrame:
        """
        Generate a sample dataset of true label values along with predicted values for these. This dataframe
        is created so as to have a precision and recall equal to those provided in the parameters, and will
        be of length n.

        Args:
            prec: Precision to use when generating the sample predictions
            rec: Recall to use when generating the sample predictions
            n: Size of sample
            p: Proportion of 0s and 1s in the sample

        Returns:
            Dataframe of length n with two columns: one holding the 'actuals' for the data, i.e. the true class,
            and the other column containing the predicted values for these, which were chosen specifically to obtain
            a recall and precision approx. equal to those passed in as arguments
        """
        if p is None:
            p = [0.86, 0.14]

        actuals = np.random.choice([0, 1], size=n, p=p)
        preds = actuals.copy()
        n_pos = np.sum(actuals)
        rec_change = (1 - rec) * n_pos

        j = 0
        for n, i in enumerate(actuals):
            if i == 1:
                preds[n] = 0
                j += 1
            if j >= rec_change:
                break

        tp = (rec * n_pos)
        prec_change = (tp - (prec * tp)) / prec

        j = 0
        for n, i in enumerate(actuals):
            if j >= prec_change:
                break

            if i == 0:
                preds[n] = 1
                j += 1

        df = pd.DataFrame({'true': actuals, 'pred': preds})
        return df
