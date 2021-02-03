import unittest
import pandas as pd
import numpy as np
from mridle.experiment import ModelRun, PartitionedExperiment
from mridle.experiments.harvey import HarveyModel
from sklearn.ensemble import RandomForestClassifier


class TestExperiment(unittest.TestCase):

    def test_integration_test(self):
        #columns = ['historic_no_show_cnt', 'days_sched_in_advance', 'sched_for_hour', 'no_show_before']
        cols_for_modeling = ['no_show_before', 'no_show_before_sq', 'sched_days_advanced', 'hour_sched',
                     'distance_to_usz', 'age', 'close_to_usz', 'male', 'female', 'age_sq',
                     'sched_days_advanced_sq', 'distance_to_usz_sq', 'sched_2_days', 'age_20_60']
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(columns))), columns=columns)
        df['noshow'] = np.where(df['historic_no_show_cnt'] > 50, 1, 0)
        harvey_model_run = HarveyModel

        exp = PartitionedExperiment(name='test', data_set=df, label_key='noshow', preprocessing_func=None,
                                    model_run_class=harvey_model_run, model=RandomForestClassifier(), hyperparams={},
                                    verbose=True)
        results = exp.run(run_hyperparam_search=False)
        print(results)
        print("Evaluation")
        print(exp.show_evaluation())
        print("Feature Importances")
        print(exp.show_feature_importances())

        self.assertTrue(results is not None)
