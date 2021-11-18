import unittest
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from mridle.experiment.architecture import ArchitectureInterface


def get_test_data_set():
    column_names = ['A', 'B', 'C', 'D']
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
    df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
    return df


class TestArchitectureInterface(unittest.TestCase):

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
            'name': 'cyc',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing.StandardScaler',
                        'name': 'scaler',
                        'args': {'columns': ['A', 'B']},
                        'config': {'with_mean': True, },
                    },
                    {
                        'flavor': 'sklearn.svm.SVC',
                        'name': 'svc',
                        'args': {'columns': ['C', 'D']},
                        'config': {},
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

    def test_pipeline_serialize(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

        expected_serialization = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.preprocessing._data.StandardScaler',
                        'name': 'scaler',
                        'config': {
                            'copy': True,
                            'with_mean': True,
                            'with_std': True
                        }
                    },
                    {
                        'flavor': 'sklearn.svm._classes.SVC',
                        'name': 'svc',
                        'config': {
                            'C': 1.0,
                            'break_ties': False,
                            'cache_size': 200,
                            'class_weight': None,
                            'coef0': 0.0,
                            'decision_function_shape': 'ovr',
                            'degree': 3,
                            'gamma': 'scale',
                            'kernel': 'rbf',
                            'max_iter': -1,
                            'probability': False,
                            'random_state': None,
                            'shrinking': True,
                            'tol': 0.001,
                            'verbose': False
                        }
                    },
                ],
            },
        }
        serialization = ArchitectureInterface.serialize(pipe)
        self.assertEqual(serialization, expected_serialization)

    def test_nested_pipeline_serialize(self):
        column_names = ['A', 'B', 'C', 'D']
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
        df['E'] = random.choices(['one', 'two', 'three'], k=100)
        df['label'] = np.where(df[column_names[0]] > 50, 1, 0)

        categorical_columns = ['E']
        numerical_columns = ['A', 'B', 'C', 'D']

        categorical_encoder = OneHotEncoder(handle_unknown='ignore')

        numerical_pipe = Pipeline([
            ('scaler', StandardScaler())
        ])

        preprocessing = ColumnTransformer(
            [('cat', categorical_encoder, categorical_columns),
             ('num', numerical_pipe, numerical_columns)
             ])

        lr_pipe = Pipeline([
            ('preprocess', preprocessing),
            ('classifier',
             LogisticRegression(random_state=94, penalty='l1', solver='liblinear', class_weight='balanced'))
        ])

        # x = df.copy().drop('label', axis=1)
        # y = df['label']
        # lr_pipe.fit(x, y)

        expected_serialization = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': [
                    {
                        'flavor': 'sklearn.compose.ColumnTransformer',
                        'name': 'preprocess',
                        'config': {
                            'steps': [
                                {
                                    'flavor': 'sklearn.preprocessing._encoders.OneHotEncoder',
                                    'name': 'cat',
                                    'args': {'columns': ['E']},
                                    'config': {
                                        'categories': 'auto',
                                        'drop': None,
                                        # 'dtype': 'numpy.float64',  # TODO: this is a problem
                                        'handle_unknown': 'ignore',
                                        'sparse': True
                                    },
                                },
                                {
                                    'flavor': 'sklearn.pipeline.Pipeline',
                                    'name': 'num',
                                    'args': {'columns': ['A', 'B', 'C', 'D']},
                                    'config': {
                                        'steps': [
                                            {
                                                'flavor': 'sklearn.preprocessing._data.StandardScaler',
                                                'name': 'scaler',
                                                'config': {
                                                    'copy': True,
                                                    'with_mean': True,
                                                    'with_std': True
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        },
                    },
                    {
                        'flavor': 'sklearn.linear_model._logistic.LogisticRegression',
                        'name': 'classifier',
                        'config': {
                            'C': 1.0,
                            'class_weight': 'balanced',
                            'dual': False,
                            'fit_intercept': True,
                            'intercept_scaling': 1,
                            'l1_ratio': None,
                            'max_iter': 100,
                            'multi_class': 'auto',
                            'n_jobs': None,
                            'penalty': 'l1',
                            'random_state': 94,
                            'solver': 'liblinear',
                            'tol': 0.0001,
                            'verbose': 0,
                            'warm_start': False
                        },
                    }
                ]
            },
        }
        serialization = ArchitectureInterface.serialize(lr_pipe)
        self.assertEqual(serialization, expected_serialization)

    def test_nested_pipeline_serialize_deserialize(self):
        column_names = ['A', 'B', 'C', 'D']
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(column_names))), columns=column_names)
        df['E'] = random.choices(['one', 'two', 'three'], k=100)
        df['label'] = np.where(df[column_names[0]] > 50, 1, 0)
        x = df.copy().drop('label', axis=1)
        y = df['label']

        categorical_columns = ['E']
        numerical_columns = ['A', 'B', 'C', 'D']

        categorical_encoder = OneHotEncoder(handle_unknown='ignore')  # 0a

        numerical_pipe = Pipeline([  # 0b
            ('scaler', StandardScaler())  # 0ba
        ])

        preprocessing = ColumnTransformer(
            [('cat', categorical_encoder, categorical_columns),  # 0a
             ('num', numerical_pipe, numerical_columns)  # 0b
             ])

        log_reg = LogisticRegression(random_state=94, penalty='l1', solver='liblinear', class_weight='balanced')
        pipe = Pipeline([
            ('preprocess', preprocessing),  # 0
            ('classifier', log_reg),  # 1

        ])

        serialization = ArchitectureInterface.serialize(pipe)
        recreated_pipe = ArchitectureInterface.deserialize(serialization)

        self.assertEqual(type(recreated_pipe), Pipeline)
        self.assertEqual(len(recreated_pipe.steps), 2)

        step_0_name, step_0_obj = recreated_pipe.steps[0]
        self.assertEqual(step_0_name, 'preprocess')
        self.assertEqual(type(step_0_obj), ColumnTransformer)

        step_0a_name, step_0a_obj, step_0a_cols = step_0_obj.transformers[0]
        self.assertEqual(step_0a_name, 'cat')
        self.assertEqual(type(step_0a_obj), OneHotEncoder)
        self.assertEqual(step_0a_cols, categorical_columns)
        self.assertEqual(step_0a_obj.get_params(), categorical_encoder.get_params())

        step_0b_name, step_0b_obj, step_0b_cols = step_0_obj.transformers[1]
        self.assertEqual(step_0b_name, 'num')
        self.assertEqual(type(step_0b_obj), Pipeline)
        self.assertEqual(step_0b_cols, numerical_columns)

        step_0ba_name, step_0ba_obj = step_0b_obj.steps[0]
        self.assertEqual(step_0ba_name, 'scaler')
        self.assertEqual(type(step_0ba_obj), StandardScaler)

        step_1_name, step_1_obj = recreated_pipe.steps[1]
        self.assertEqual(step_1_name, 'classifier')
        self.assertEqual(type(step_1_obj), LogisticRegression)
        self.assertEqual(step_1_obj.get_params(), log_reg.get_params())

        pipe.fit(x, y)
        original_pipe_result = pipe.predict(x)

        recreated_pipe.fit(x, y)
        recreated_pipe_result = recreated_pipe.predict(x)

        np.testing.assert_array_almost_equal(recreated_pipe_result, original_pipe_result)