import unittest
from mridle.experiment import PartitionedExperiment
from mridle.experiments.harvey import HarveyModel
from sklearn.ensemble import RandomForestClassifier


class TestExperiment(unittest.TestCase):

    def test_integration_test(self):
        harvey_model_run = HarveyModel
        data_set = harvey_model_run.get_test_data_set()
        label_key = 'noshow'

        exp = PartitionedExperiment(name='test', data_set=data_set, label_key=label_key, preprocessing_func=None,
                                    model_run_class=harvey_model_run, model=RandomForestClassifier(), hyperparams={},
                                    verbose=True)
        results = exp.run(run_hyperparam_search=False)
        print(results)
        print("Evaluation")
        print(exp.show_evaluation())
        print("Feature Importances")
        print(exp.show_feature_importances())

        self.assertTrue(results is not None)

    def test_generate_file_name(self):
        data_set = HarveyModel.get_test_data_set()
        label_key = 'noshow'
        model = RandomForestClassifier()
        harvey_model_run = HarveyModel(train_set=data_set, test_set=data_set, label_key=label_key, hyperparams={},
                                       model=model, preprocessing_func=None)

        file_name = harvey_model_run.generate_file_name()
        expected_file_name = '0000-00-00_00-00-00__RandomForestClassifier.pkl'
        file_name_parts = file_name.split('__')
        expected_file_name_parts = expected_file_name.split('__')
        self.assertEqual(len(file_name_parts), len(expected_file_name_parts))
        self.assertEqual(file_name_parts[1], expected_file_name_parts[1])

    def test_generate_file_name_with_descriptor(self):
        data_set = HarveyModel.get_test_data_set()
        label_key = 'noshow'
        model = RandomForestClassifier()
        harvey_model_run = HarveyModel(train_set=data_set, test_set=data_set, label_key=label_key, hyperparams={},
                                       model=model, preprocessing_func=None)

        file_name = harvey_model_run.generate_file_name('5-features')
        expected_file_name = '0000-00-00_00-00-00__RandomForestClassifier__5-features.pkl'
        file_name_parts = file_name.split('__')
        expected_file_name_parts = expected_file_name.split('__')
        self.assertEqual(len(file_name_parts), len(expected_file_name_parts))
        self.assertEqual(file_name_parts[1], expected_file_name_parts[1])
