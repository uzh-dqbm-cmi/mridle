import unittest
import pandas as pd
import numpy as np
from mridle.utilities.experiment import PartitionedExperiment, ModelRun
from sklearn.ensemble import RandomForestClassifier


def get_test_data_set():
    column_names = ['A', 'B', 'C', 'D']
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
    df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
    return df


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.df = get_test_data_set()
        self.train_df = self.df[:60]
        self.test_df = self.df[60:]

    def test_base_ModelRun_integration_test(self):
        exp = PartitionedExperiment(name='test', data_set=self.df, feature_subset=list('ABCD'),
                                    label_key='label', preprocessing_func=None,
                                    model_run_class=ModelRun, model=RandomForestClassifier(), hyperparams={},
                                    verbose=True, search_type='random', num_cv_folds=2, num_iter=2,
                                    scoring_fn='f1_score')
        results = exp.run(run_hyperparam_search=False)
        print(results)
        print("Evaluation")
        print(exp.show_evaluation())
        print("Feature Importances")
        print(exp.show_feature_importances())

        self.assertTrue(results is not None)

    # TODO: move to src/tests/pipelines/data_science/test_harvey.py
    # def test_harvey_integration_test(self):
    #     harvey_model_run = HarveyModel
    #     data_set = harvey_model_run.get_test_data_set()
    #     label_key = 'noshow'
    #     feature_subset = ['no_show_before', 'no_show_before_sq', 'sched_days_advanced', 'hour_sched',
    #                       'distance_to_usz', 'age', 'close_to_usz']
    #     exp = PartitionedExperiment(name='test', data_set=data_set, feature_subset=feature_subset, label_key=label_key,
    #                                 preprocessing_func=None, model_run_class=harvey_model_run,
    #                                 model=RandomForestClassifier(), hyperparams={},
    #                                 verbose=True, search_type='random', num_cv_folds=2, num_iter=2,
    #                                 scoring_fn='f1_score')
    #     results = exp.run(run_hyperparam_search=False)
    #     print(results)
    #     print("Evaluation")
    #     print(exp.show_evaluation())
    #     print("Feature Importances")
    #     print(exp.show_feature_importances())
    #
    #     self.assertTrue(results is not None)

    def test_generate_file_name(self):
        model = RandomForestClassifier()
        harvey_model_run = ModelRun(train_set=self.train_df, test_set=self.test_df, label_key='label', hyperparams={},
                                    model=model, preprocessing_func=None, search_type='random', num_cv_folds=2,
                                    num_iter=2, scoring_fn='f1_score')

        file_name = harvey_model_run.generate_file_name()
        expected_file_name = '0000-00-00_00-00-00__RandomForestClassifier.pkl'
        file_name_parts = file_name.split('__')
        expected_file_name_parts = expected_file_name.split('__')
        self.assertEqual(len(file_name_parts), len(expected_file_name_parts))
        self.assertEqual(file_name_parts[1], expected_file_name_parts[1])

    def test_generate_file_name_with_descriptor(self):
        model = RandomForestClassifier()
        harvey_model_run = ModelRun(train_set=self.train_df, test_set=self.test_df, label_key='label', hyperparams={},
                                    model=model, preprocessing_func=None, search_type='random', num_cv_folds=2,
                                    num_iter=2, scoring_fn='f1_score')

        file_name = harvey_model_run.generate_file_name('5-features')
        expected_file_name = '0000-00-00_00-00-00__RandomForestClassifier__5-features.pkl'
        file_name_parts = file_name.split('__')
        expected_file_name_parts = expected_file_name.split('__')
        self.assertEqual(len(file_name_parts), len(expected_file_name_parts))
        self.assertEqual(file_name_parts[1], expected_file_name_parts[1])

    def test_generate_file_name_with_descriptor_with_spaces(self):
        model = RandomForestClassifier()
        harvey_model_run = ModelRun(train_set=self.train_df, test_set=self.test_df, label_key='label', hyperparams={},
                                    model=model, preprocessing_func=None, search_type='random', num_cv_folds=2,
                                    num_iter=2, scoring_fn='f1_score')

        file_name = harvey_model_run.generate_file_name('5 features')
        expected_file_name = '0000-00-00_00-00-00__RandomForestClassifier__5-features.pkl'
        file_name_parts = file_name.split('__')
        expected_file_name_parts = expected_file_name.split('__')
        self.assertEqual(len(file_name_parts), len(expected_file_name_parts))
        self.assertEqual(file_name_parts[1], expected_file_name_parts[1])
