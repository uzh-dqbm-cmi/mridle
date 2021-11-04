import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp
from mridle.experiment.experiment import Experiment
from mridle.experiment.dataset import DataSet
from mridle.experiment.stratifier import Stratifier, PartitionedLabelStratifier, TrainTestStratifier  # noqa
from mridle.experiment.trainer import Trainer, SkorchTrainer  # noqa
from mridle.experiment.tuner import Tuner, RandomSearchTuner, BayesianTuner  # noqa
from mridle.experiment.metric import Metric, F1_Macro, AUPRC, AUROC, LogLoss  # noqa


def test_config():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
    df = df.dropna().copy()

    exp_config = {
        'DataSet': {
            'flavor': 'DataSet',
            'config': {
                'features': [
                    'pclass',
                    'age',
                    'adult_male',
                    'sibsp',
                    'parch'
                ],
                'target': 'survived',
            },
        },
        'Stratifier': {
            'flavor': 'PartitionedLabelStratifier',
            'config': {
                'n_partitions': 3,
            },
        },
        'Architecture': {
            'flavor': 'RandomForestClassifier'
        },
        'Trainer': {
            'flavor': 'Trainer',
            'config': {
                # 'epochs': 10,
            }
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
        'Metric': [
            {'flavor': 'F1_Macro', 'config': {'classification_cutoff': 0.5}},
            {'flavor': 'AUPRC'},
            {'flavor': 'AUROC'},
            {'flavor': 'LogLoss'},
        ]
    }

    exp = Experiment.configure(config=exp_config, data=df)
    results = exp.go()
    print(results)


def ex():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
    df = df.dropna().copy()

    config = {
        'DataSet': {
            'features': [
                'pclass',
                'age',
                'adult_male',
                'sibsp',
                'parch'
            ],
            'target': 'survived',
        },
        'Stratifier': {
            'n_partitions': 1,
            'test_split_size': 0.2
        },
        'Trainer': {
            # 'epochs': 10,
        },
        'Tuner': {
            'hyperparameters': {
                'n_estimators': hp.choice('n_estimators', [10, 100, 200]),
                'max_depth': hp.choice('max_depth', [2, 4, 10, 50, 100, None]),
                'min_samples_split': hp.choice('min_samples_split', [2, 4, 6, 8, 10]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10]),
            },
            'num_cv_folds': 3,
            'num_iters': 100,
            'scoring_function': 'auprc',
            'verbose': 1,
            'hyperopt_timeout': 5 * 60
        },
    }
    data_set = DataSet(df, config['DataSet'])
    stratifier = TrainTestStratifier(config['Stratifier'])
    architecture = RandomForestClassifier()
    tuner = BayesianTuner(config['Tuner'])
    trainer = Trainer(architecture, config['Trainer'], tuner)
    # trainer = RandomForestClassifier()

    metrics = [F1_Macro(config={'classification_cutoff': 0.5}), AUPRC(), AUROC(), LogLoss()]

    exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics)
    results = exp.go()
    print(results)


if 0:
    print("Running test experiment")
    ex()

if 0:
    print("Running configuration experiment")
    test_config()
