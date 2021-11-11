import unittest
import pandas as pd
import numpy as np
from mridle.experiment.experiment import Experiment
from mridle.experiment.dataset import DataSet
from mridle.experiment.stratifier import TrainTestStratifier
from mridle.experiment.architecture import ArchitectureInterface
from mridle.experiment.trainer import Trainer
from mridle.experiment.tuner import RandomSearchTuner
from mridle.experiment.metric import Metric
from sklearn.ensemble import RandomForestClassifier


def get_test_data_set():
    column_names = ['A', 'B', 'C', 'D']
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
    df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
    return df


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.df = get_test_data_set()
        self.configuration = {
            'DataSet': {
                'flavor': 'DataSet',
                'config': {
                    'features': ['A', 'B', 'C', 'D'],
                    'target': 'label',
                },
            },
            'Stratifier': {
                'flavor': 'TrainTestStratifier',
                'config': {
                    'test_split_size': 0.3,
                },
            },
            'Architecture': {
                'flavor': 'RandomForestClassifier',
                'config': {}
            },
            'Trainer': {
                'flavor': 'Trainer',
                'config': {}
            },
            'Tuner': {
                'flavor': 'RandomSearchTuner',
                'config': {
                    'hyperparameters': {
                        'n_estimators': range(200, 2000, 10),
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': range(10, 110, 11),
                        'min_samples_split': [2, 4, 6, 8, 10],
                        'min_samples_leaf': [1, 2, 5, 10],
                        'bootstrap': [True, False],
                    },
                    'num_cv_folds': 3,
                    'num_iters': 5,
                    'scoring_function': 'f1_macro',
                    'verbose': 1,
                },
            },
            'Metrics': [
                {'flavor': 'F1_Macro', 'config': {'classification_cutoff': 0.5}},
                {'flavor': 'AUPRC', 'config': {}},
                {'flavor': 'AUROC', 'config': {}},
                {'flavor': 'LogLoss', 'config': {}},
            ]
        }
        self.configuration_without_tuner = {
            'DataSet': {
                'flavor': 'DataSet',
                'config': {
                    'features': ['A', 'B', 'C', 'D'],
                    'target': 'label',
                },
            },
            'Stratifier': {
                'flavor': 'TrainTestStratifier',
                'config': {
                    'test_split_size': 0.3,
                },
            },
            'Architecture': {
                'flavor': 'RandomForestClassifier',
                'config': {}
            },
            'Trainer': {
                'flavor': 'Trainer',
                'config': {}
            },
            'Metrics': [
                {'flavor': 'F1_Macro', 'config': {'classification_cutoff': 0.5}},
                {'flavor': 'AUPRC', 'config': {}},
                {'flavor': 'AUROC', 'config': {}},
                {'flavor': 'LogLoss', 'config': {}},
            ]
        }

    def test_configure(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)

        self.assertTrue(isinstance(exp.dataset, DataSet))
        self.assertEqual(exp.dataset.config, self.configuration['DataSet']['config'])
        pd.testing.assert_frame_equal(exp.dataset.data, self.df)

        self.assertTrue(isinstance(exp.stratifier, TrainTestStratifier))
        self.assertEqual(exp.stratifier.config, self.configuration['Stratifier']['config'])

        self.assertTrue(isinstance(exp.trainer.architecture, RandomForestClassifier))
        # self.assertEqual(exp.dataset.config, self.configuration['DataSet']['config'])  # TODO How to test this?

        self.assertTrue(isinstance(exp.trainer, Trainer))
        self.assertEqual(exp.trainer.config, self.configuration['Trainer']['config'])

        self.assertTrue(isinstance(exp.trainer.tuner, RandomSearchTuner))
        self.assertEqual(exp.trainer.tuner.config, self.configuration['Tuner']['config'])

        self.assertTrue(isinstance(exp.metrics, list))
        for i, metric in enumerate(exp.metrics):
            self.assertTrue(isinstance(metric, Metric))
            self.assertEqual(metric.config, self.configuration['Metrics'][i]['config'])

    def test_serialize(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)
        exp_dict = exp.serialize()
        for expected_key in ['metadata', 'components']:
            self.assertTrue(expected_key in exp_dict)

        for component in ['DataSet', 'Stratifier', 'Architecture', 'Trainer', 'Tuner', 'Metrics']:
            component_dict = exp_dict['components'][component]
            orig_component_dict = self.configuration[component]

            if component == 'Architecture':
                continue  # TODO

            if component == 'Metrics':
                for i, m in enumerate(component_dict):
                    for key in ['flavor', 'config']:
                        self.assertEqual(component_dict[i][key], orig_component_dict[i][key])
            else:
                for key in ['flavor', 'config']:
                    self.assertEqual(component_dict[key], orig_component_dict[key])

    def test_deserialize(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)
        exp_dict = exp.serialize()
        exp_deserialized = Experiment.deserialize(exp_dict)

        # DataSet
        self.assertEqual(type(exp_deserialized.dataset), type(exp.dataset))
        pd.testing.assert_frame_equal(exp_deserialized.dataset.data, exp.dataset.data)

        # Stratifier
        self.assertEqual(type(exp_deserialized.stratifier), type(exp.stratifier))
        self.assertEqual(exp_deserialized.stratifier.config, exp.stratifier.config)
        for p, partition in enumerate(exp.stratifier.partition_idxs):
            for d, indx in enumerate(partition):
                np.testing.assert_almost_equal(exp_deserialized.stratifier.partition_idxs[p][d],
                                               exp.stratifier.partition_idxs[p][d])
                # for i, v in enumerate(indx):
                #     self.assertEqual(exp_deserialized.stratifier.partition_idxs[p][d][i],
                #                      exp.stratifier.partition_idxs[p][d][i])

        # Trainer
        self.assertEqual(type(exp_deserialized.trainer), type(exp.trainer))
        self.assertEqual(exp_deserialized.trainer.config, exp.trainer.config)

        # Architecture
        # TODO architecture

        # Tuner
        self.assertEqual(type(exp_deserialized.trainer.tuner), type(exp.trainer.tuner))
        self.assertEqual(exp_deserialized.trainer.tuner.config, exp.trainer.tuner.config)

        # Metrics
        for i, metric in enumerate(exp.metrics):
            self.assertEqual(type(exp_deserialized.metrics[i]), type(exp.metrics[i]))
            self.assertEqual(exp_deserialized.metrics[i].config, exp.metrics[i].config)

        # Metadata
        self.assertEqual(exp_deserialized.run_date, exp.run_date)

    def test_serialize_deserialize_without_Tuner(self):
        exp = Experiment.configure(config=self.configuration_without_tuner, data=self.df)
        exp_dict = exp.serialize()

        self.assertTrue('Tuner' not in exp_dict['components'])
        exp_deserialized = Experiment.deserialize(exp_dict)

        # Tuner
        self.assertIsNone(exp_deserialized.trainer.tuner)

    def test_deserialize_after_go(self):
        exp = Experiment.configure(config=self.configuration, data=self.df)
        exp.go()
        exp_dict = exp.serialize()
        exp_deserialized = Experiment.deserialize(exp_dict)

        # DataSet
        self.assertEqual(type(exp_deserialized.dataset), type(exp.dataset))
        pd.testing.assert_frame_equal(exp_deserialized.dataset.data, exp.dataset.data)

        # Stratifier
        self.assertEqual(type(exp_deserialized.stratifier), type(exp.stratifier))
        self.assertEqual(exp_deserialized.stratifier.config, exp.stratifier.config)
        for p, partition in enumerate(exp.stratifier.partition_idxs):
            for d, indx in enumerate(partition):
                np.testing.assert_almost_equal(exp_deserialized.stratifier.partition_idxs[p][d],
                                               exp.stratifier.partition_idxs[p][d])
                # for i, v in enumerate(indx):
                #     self.assertEqual(exp_deserialized.stratifier.partition_idxs[p][d][i],
                #                      exp.stratifier.partition_idxs[p][d][i])

        # Trainer
        self.assertEqual(type(exp_deserialized.trainer), type(exp.trainer))
        self.assertEqual(exp_deserialized.trainer.config, exp.trainer.config)

        # Architecture
        # TODO architecture

        # Tuner
        self.assertEqual(type(exp_deserialized.trainer.tuner), type(exp.trainer.tuner))
        self.assertEqual(exp_deserialized.trainer.tuner.config, exp.trainer.tuner.config)

        # Metrics
        for i, metric in enumerate(exp.metrics):
            self.assertEqual(type(exp_deserialized.metrics[i]), type(exp.metrics[i]))
            self.assertEqual(exp_deserialized.metrics[i].config, exp.metrics[i].config)

        # Metadata
        self.assertEqual(exp_deserialized.run_date, exp.run_date)

        # Evaluation
        pd.testing.assert_frame_equal(exp_deserialized.evaluation, exp.evaluation)

        # Predictors
        # assert that for every partition, the predictors predict the same values
        for partition_idx in range(exp.stratifier.n_partitions):
            x_train, y_train, x_test, y_test = exp.stratifier.materialize_partition(partition_idx)
            x_train_des, y_train_des, x_test_des, y_test_des = exp.stratifier.materialize_partition(partition_idx)
            # before checking the predictors, make sure the X they're predicting on is the same.
            pd.testing.assert_frame_equal(x_test_des, x_test)
            y_pred = exp.partition_predictors[partition_idx].predict(x_test)
            y_pred_des = exp_deserialized.partition_predictors[partition_idx].predict(x_test_des)
            np.testing.assert_almost_equal(y_pred_des, y_pred)

    @staticmethod
    def configure(d):
        from sklearn.pipeline import Pipeline
        step_list = []
        for step in d['config']['steps']:
            step_obj = ArchitectureInterface.configure(step)
        return Pipeline(step_list)

    def test_pipeline_configure(self):
        config = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing.StandardScaler',
                        'config': {
                            'with_mean': True,
                        }
                    },
                    {
                        'flavor': 'sklearn.svm.SVC',
                        'config': {

                        }
                    },
                ],
            },
        }
        pipe = ArchitectureInterface.configure(config)

    def test_nested_pipeline_configure(self):
        one_pipe_config = {
            'flavor': 'sklearn.compose.ColumnTransformer',
            'config': {
                'name': 'cyc',
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing.StandardScaler',
                        'config': {
                            'with_mean': True,
                        },
                    },
                    {
                        'flavor': 'sklearn.svm.SVC',
                        'config': {

                        },
                    },
                ],
                'args': [
                    {
                        'columns': [],
                    },
                ],
            },
        }
        nested_config = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps':
                    [
                        one_pipe_config,
                    ],
            },
        }
        pipe = ArchitectureInterface.configure(nested_config)

    # def test_parse_pipeline_params(self):
    #     from sklearn.svm import SVC
    #     from sklearn.preprocessing import StandardScaler
    #     from sklearn.pipeline import Pipeline
    #     pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    #
    #
    #
    #     params = {
    #         'memory': None,
    #         'steps': [('standardscaler', StandardScaler()), ('gaussiannb', GaussianNB())],
    #         'verbose': False,
    #         'standardscaler': StandardScaler(),
    #         'gaussiannb': GaussianNB(),
    #         'standardscaler__copy': True,
    #         'standardscaler__with_mean': True,
    #         'standardscaler__with_std': True,
    #         'gaussiannb__priors': None,
    #         'gaussiannb__var_smoothing': 1e-09
    #     }
    #
    #     for object in params['steps']:
    #         object_params =
