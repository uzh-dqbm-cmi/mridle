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
