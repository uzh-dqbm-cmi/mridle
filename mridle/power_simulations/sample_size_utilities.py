import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from multiprocessing import Pool, Queue
import itertools
import logging
from logging.handlers import QueueHandler, QueueListener
from typing import List, Tuple
from pathlib import Path
import pickle
import datetime
from sympy import Eq, solve
from sympy.abc import a, b, c, d


class PowerSimulations:
    """
    Class for running power simulations for calculating the sample size, effect size, and power for the
    'silent live test'. A list of sample sizes and a list of effect sizes are declared when creating this
    object. These two lists are then used to calculate the power of our permutation test for each combination
    of elements in these two lists.
    """

    def __init__(self, sample_sizes: List[int], effect_sizes: List[float], num_trials_per_run: int,
                 num_runs_for_power_calc: int, original_test_set_length: int, significance_level: float,
                 base_performance: float, performance_type: str, num_cpus: int, random_seed: int = None):
        """
        Create a PowerSimulations objects.

        The num_trials_per_run attribute is related to how many times we resample from our generated dataset, so as to
        create the distribution under the null hypothesis, and then we compare the 'original' test statistic from
        the given dataset to this null hypothesis to get an alpha value, and then we can reject this test or not.
        We then run this experiment num_runs_for_power_calc times (each with a newly generated dataset) to get a %
        of times that the null hypothesis was correctly rejected, giving us the power.

        Args:
            sample_sizes: List of sample sizes to calculate the power for
            effect_sizes: List of effect sizes to calculate the power for
            num_trials_per_run: number of simulations to run for the permutation test to obtain the distribution of
            the null hypothesis (1000+ recommended if computer speed allows)
            num_runs_for_power_calc: number of times to run the permutation experiment to obtain an estimate for the
            power of the test
            original_test_set_length: Number of samples to create for the 'original' test set
            significance_level: Significance level to use for the permutation test
            base_precision: Precision of the model on the original test set
            base_f1_macro: F1 macro metric for the model on the original test set
            num_cpus: Number of cpus to use for the experiment executions
        """
        self.sample_sizes = sample_sizes
        self.effect_sizes = effect_sizes
        self.num_trials_per_run = num_trials_per_run
        self.num_runs_for_power_calc = num_runs_for_power_calc
        self.original_test_set_length = original_test_set_length
        self.significance_level = np.float64(significance_level)
        self.base_performance = base_performance
        self.performance_type = performance_type
        self.results = None
        self.num_cpus = num_cpus
        self.random_seed = random_seed
        self.run_id = ''

    def run(self, log_to_file: bool = True):
        """
        Run the permutation tests and save the results as an attribute of the object. The results are saved as a
        dataframe, with three columns:
            - effect_size: effect size that the test was performed for. A specific value from the list of values
            declared at object initialisation
            - sample_size: sample size that the test was performed for. A specific value from the list of values
            declared at object initialisation
            - power: Resulting power of the test, using the effect and sample size given.

        Args:
            log_to_file: whether to log the progress to a file (otherwise, logs go to stdout).
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_id = f'power_simulations_{timestamp}'
        self.run_id = run_id
        q_listener, q = logger_init(log_name=run_id, log_to_file=log_to_file)
        self.log_initial_values()

        effect_sample_sizes = list(itertools.product(self.effect_sizes, self.sample_sizes))
        with Pool(self.num_cpus, worker_init, [q]) as p:
            power_results = p.map(self.run_simulation_for_effect_size_sample_size, effect_sample_sizes)

        results_df = pd.DataFrame(effect_sample_sizes, power_results)
        results_df.reset_index(inplace=True)
        results_df.columns = ['power', 'effect_size', 'sample_size']
        results_df = results_df[['effect_size', 'sample_size', 'power']]
        self.results = results_df
        q_listener.stop()

    def log_initial_values(self):
        logging.info(f'sample_sizes: {self.sample_sizes}')
        logging.info(f'effect_sizes: {self.effect_sizes}')
        logging.info(f'num_trials_per_run: {self.num_trials_per_run}')
        logging.info(f'num_runs_for_power_calc: {self.num_runs_for_power_calc}')
        logging.info(f'original_test_set_length: {self.original_test_set_length}')
        logging.info(f'significance_level: {self.significance_level}')
        logging.info(f'base_performance: {self.base_performance}')
        logging.info(f'performance_type: {self.performance_type}')

    def run_simulation_for_effect_size_sample_size(self, effect_sample_sizes: Tuple[float, int]) -> List[float]:
        """
        Helper function for running the permutation experiments, helping with parallelisation.

        Args:
            effect_sample_sizes: Tuple containing a single combination of the declared effect and sample sizes

        Returns:
            List of alpha values for each test in the power analysis trials

        """
        effect_size, sample_size = effect_sample_sizes

        performance_new = self.base_performance * (1 - effect_size)

        if self.random_seed:
            np.random.seed(self.random_seed)

        alphas = [self.run_permutation_trials(performance_new, sample_size)
                  for i in range(self.num_runs_for_power_calc)]
        power = np.sum(alphas < self.significance_level) / len(alphas)

        logging.info(f'Completed permutation: effect & sample size:{effect_sample_sizes}; Power:{power}')

        return power

    def run_permutation_trials(self, performance_new: float, sample_size_new: int) -> float:
        """
        Execute n=self.num_trials_per_run runs of the individual permuted trial in the function one_trial.
        An alpha value for this trial is returned, and we will then obtain n=self.num_runs_for_power_calc values
        for this alpha value by running this function over and over again. Using this set of alpha values, we
        can then calculated the power of our test.

        Args:
            performance_new
            sample_size_new: Sample size of new test set to be generated

        Returns:
            Alpha value for one group of permutation tests.

        """

        df = self.generate_actuals_preds(self.base_performance, self.original_test_set_length)
        df_new = self.generate_actuals_preds(performance_new, sample_size_new)

        pooled = pd.concat([df, df_new])
        orig_diff = f1_score(df['true'], df['pred'], average='macro') - f1_score(df_new['true'], df_new['pred'],
                                                                                 average='macro')

        differences = [self.run_single_trial(pooled) for i in range(self.num_trials_per_run)]
        individual_alpha = np.sum(differences > orig_diff) / len(differences)
        return individual_alpha

    def run_single_trial(self, pooled_data: pd.DataFrame) -> float:
        """
        Take in pooled data, create one set of permuted datasets and return the test statistic for this one trial
        Args:
            pooled_data: Both the 'original' test set and the new test set combined into one dataframe, which
            we then use to split into two random groups of the same size as each of the two individual datasets

        Returns:
            Test statistic for one run of the permutation trials

        """
        new_orig_df, new_new_df = permute_and_split(pooled_data, split_point=self.original_test_set_length)
        score = calculate_f1_diff(new_orig_df, new_new_df)

        return score

    def generate_actuals_preds(self, performance: float, n: int) -> pd.DataFrame:
        """

        Returns:

        """
        if self.performance_type == 'precision':
            return self.generate_actuals_preds_precision(performance, n)
        elif self.performance_type == 'f1_macro':
            return self.generate_actuals_preds_f1_macro(performance, n)

    @staticmethod
    def generate_actuals_preds_precision(performance: float, n: int, p: List[float] = None) -> pd.DataFrame:
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

        # We're focusing on precision, but need a value for recall as well. So we just set this to be equal to precision
        prec = performance
        rec = performance

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

    @staticmethod
    def generate_actuals_preds_f1_macro(f1: float, n: int, p=None) -> pd.DataFrame:
        """

        Args:
            f1:
            n:
            p:

        Returns:

        """
        if p is None:
            p = [0.86, 0.14]
        wanted_f1 = f1
        recall_0 = wanted_f1 * 1.1 if wanted_f1 * 1.1 < 1 else wanted_f1
        n0, n1 = p[0] * n, p[1] * n

        # c = tp0, d = tp1, a=fn0=fp1, b=fn1=fp0
        sol_found = False
        tp0 = tp1 = fn0 = fn1 = 0
        while not sol_found:
            # using solve from sympy, with each equation of form EQ(x, y), denoting x=y
            sol = solve(
                [Eq((c / (2 * c + a + b)) + (d / (2 * d + a + b)), wanted_f1),
                 Eq(c / (c + a), recall_0),
                 Eq(a + c, n0),
                 Eq(b + d, n1)]
            )

            for solution in sol:
                feasible = all(v > 0 for v in solution.values())
                if feasible:
                    tp0 = round(solution[c])
                    tp1 = round(solution[d])
                    fn0 = round(solution[a])
                    fn1 = round(solution[b])
                    sol_found = True
                else:
                    recall_0 = recall_0 * 1.01

        sol_list = [[0, 0]] * tp0
        sol_list.extend([[0, 1]] * fn0)
        sol_list.extend([[1, 1]] * tp1)
        sol_list.extend([[1, 0]] * fn1)
        solution_df = pd.DataFrame(sol_list, columns=['true', 'pred'])
        return solution_df

    def save(self, parent_directory: str, descriptor: str = '') -> Path:
        """
        Save a dictionary of the results as a pickle to a parent_directory with the same filename as the logging
        file (different file extension)

        Args:
            parent_directory: The parent directory in which to save the model.
            descriptor: Optional short string to append to filename to easily see which trial the results
            correspond with


        Returns: File path of the saved object.

        Example Usage:
            >>> power_sim.save('project/results/')
            >>> # saves project/results/power_simulations_YYYY-MM-DD_HH-MM-SS.pkl
            >>> power_sim.save('project/results/', descriptor='orig_4000')
            >>> # saves project/results/power_simulations_YYYY-MM-DD_HH-MM-SS__orig_4000.pkl
        """
        # Used if/else because I wanted underscore between filename and descriptor if given, and no underscore if not
        if descriptor:
            filename = f'{self.run_id}_{descriptor}.pkl'
        else:
            filename = f'{self.run_id}.pkl'

        filepath = Path(parent_directory, filename)

        to_save = {
            'results': self.results,
            'params': {'num_trials_per_run': self.num_trials_per_run,
                       'num_runs_for_power_calc': self.num_runs_for_power_calc,
                       'original_test_set_length': self.original_test_set_length,
                       'significance_level': self.significance_level,
                       'base_performance': self.base_performance,
                       'performance_type': self.performance_type,
                       'random_seed': self.random_seed}
        }
        with open(filepath, 'wb+') as f:
            pickle.dump(to_save, f)
        return filepath


def logger_init(log_name: str, log_to_file: bool = True) -> Tuple[QueueListener, Queue]:
    """
    Initialize a logger for use in multiprocessing.

    From https://stackoverflow.com/a/34964369

    Args:
        log_name: name for log filename if log_to_file is True.
        log_to_file: Whether the logging should be saved in a file. If so, the logs will be saved to a file in the
         current working directory with filename in the style of <timestamp>_<name>.log.

    Returns: QueueListener and Queue

    """
    if log_to_file:
        filename = f'{log_name}.log'
        handler = logging.FileHandler(filename)
        print(f'Logging to file {filename}')
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(process)s - %(asctime)s - %(levelname)s: %(message)s"))

    q = Queue()
    ql = QueueListener(q, handler)
    ql.start()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # add the handler to the logger so records from this process are handled
    logger.addHandler(handler)

    return ql, q


def worker_init(q: Queue) -> None:
    """
    Initialize the logger for a child process of multiprocessing.Pool. Once this initialization is run, the child
    process just calls `logging.info`. This function relies on `logger_init` being run by the parent process first.

    From https://stackoverflow.com/a/34964369

    Args:
        q: multiprocessing.Queue for a logging QueueHandler.

    Returns:

    """
    # all records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)


def calculate_f1_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Given two dataframes, both with columns 'true' and 'pred', return the difference between the f1_score of both
    of these dataframes.

    Args:
        df1: first dataframe
        df2: second dataframe

    Returns:
        F1 score difference between the two provided dataframes of predicitons
    """
    score = f1_score(df1['true'], df1['pred'], average='macro') - f1_score(df2['true'], df2['pred'], average='macro')
    return score


def permute_and_split(pooled_data: pd.DataFrame, split_point: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Take in dataset and split it into two dataframes, using provided split point

    Args:
        pooled_data: dataframe to be split
        split_point: integer - point at which to split (i.e. row number)

    Returns:
        Two dataframes which when 'unioned' would give the original dataframe

    """
    permuted = pooled_data.sample(frac=1)
    new_orig_df = permuted[:split_point]
    new_new_df = permuted[split_point:]
    return new_orig_df, new_new_df
