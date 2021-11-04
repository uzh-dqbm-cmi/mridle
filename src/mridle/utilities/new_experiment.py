from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold  # noqa
from sklearn.metrics import brier_score_loss, log_loss, f1_score, precision_recall_curve, auc, roc_auc_score # noqa
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Union


class DataSet:
    """
    A dataset class config dictionary contains the following configurable elements:
    - features: name of the features to be used in building the features tensor
    - targets: name of the features to be used in building the targets tensor
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
        self.validate_config(config, data)
        self.data = data
        self.features_list = config['features']
        self.target = config['target']

    @property
    def x(self) -> pd.DataFrame:
        return self.data[self.features_list]

    @property
    def y(self) -> pd.Series:
        return self.data[self.target]

    def to_dict(self):
        d = {
            'data': self.data.to_dict(),  # TODO: I feel like I've had problems with this before
            'features': self.features_list,
            'target': self.target,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        data = pd.DataFrame(d['data'])
        config = d
        return cls(data, config)

    @staticmethod
    def validate_config(config, data):
        """Make sure the config aligns with the data (referenced columns exist)."""
        for key in ['features', 'target']:
            if key not in config:
                raise ValueError(f"DataSet config must contain entry '{key}'.")

        for col in config['features']:
            if col not in data.columns:
                raise ValueError(f'Feature column {col} not found in dataset.')

        if config['target'] not in data.columns:
            raise ValueError(f"Target column {config['target']} not found in dataset.")

        return True


class Stratifier(ABC):
    """
    Yield data partitions.
    """

    def __init__(self, config: Dict):
        self.validate_config(config)
        self.n_partitions = config['n_partitions']
        if 'test_split_size' in config:
            self.test_split_size = config['test_split_size']

        self.partition_idxs = config.get('partition_idxs', None)
        if 'data_set' in config:
            data_set_dict = config['data_set']
            self.data_set = DataSet.from_dict(data_set_dict)
        else:
            self.data_set = None

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Iterate over the partitions.

        Returns: X_train, y_train, X_test, y_test
        """
        if self.data_set is None:
            raise ValueError('Stratifier does not have data. Use `Stratifier.load(data_set)` before iterating.')
        if self.n < self.n_partitions:
            return_value = self.materialize_partition(self.n)
            self.n += 1
            return return_value
        else:
            raise StopIteration

    def to_dict(self):
        d = {
            'n_partitions': self.n_partitions,
            'partition_idxs': self.partition_idxs,
            'data_set': self.data_set.to_dict(),
        }
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d)

    def load_data(self, data_set: DataSet):
        self.data_set = data_set
        self.partition_idxs = self.partition_data(self.data_set)

    @abstractmethod
    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        pass

    def materialize_partition(self, partition_id) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create training and testing dataset based on the partition, which indicate the ids for the test set.

        Args:
            partition_ids: Tuple of the indices of each partition's train and test set.
            data_set: The full data set.

        Returns: X_train, y_train, X_test, y_test
        """
        train_partition_ids, test_partition_ids = self.partition_idxs[partition_id]
        X_train = self.data_set.x.iloc[train_partition_ids]
        y_train = self.data_set.y.iloc[train_partition_ids]
        X_test = self.data_set.x.iloc[test_partition_ids]
        y_test = self.data_set.y.iloc[test_partition_ids]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def validate_config(config):
        for key in ['n_partitions', ]:
            if key not in config:
                raise ValueError(f"PartitionedLabelStratifier config must contain entry '{key}'.")
        return True


class PartitionedLabelStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the doc_list into n_partitions roughly equal lists, stratified by label."""
        label_list = data_set.y
        skf = StratifiedKFold(n_splits=self.n_partitions, random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = skf.split(x, label_list)
        partitions = []
        for p_id, p in enumerate(partition_indexes):
            partitions.append(p)
        return partitions


class TrainTestStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Split data once into train and test sets. Percentage of data in test set supplied as argument."""
        df_len = len(data_set.x.index)
        perm = np.random.permutation(df_len)
        train_end = int((1-self.test_split_size) * df_len)
        train_idx = perm[:train_end]
        test_idx = perm[train_end:]
        partitions = [(train_idx, test_idx)]
        return partitions


class Architecture(ABC):

    @abstractmethod
    def fit(self, X):
        pass


class Predictor:

    def __init__(self, model=None):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        """Enforce returning only a single series."""
        y_pred_proba = self.model.predict_proba(x)
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        return y_pred_proba


class Tuner(ABC):

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def fit(self, architecture: Architecture, x, y) -> Predictor:
        pass


class Trainer:

    def __init__(self, architecture, config: Dict, tuner: Tuner = None):
        self.architecture = architecture
        self.config = config
        self.tuner = tuner

    def fit(self, x, y) -> Predictor:
        if self.tuner:
            trained_model = self.tuner.fit(self.architecture, x, y)
        else:
            trained_model = self.architecture.fit(x, y)
        return Predictor(trained_model)


class SkorchTrainer(Trainer):

    def fit(self, x, y) -> Predictor:
        y = self.transform_y(y)
        if self.tuner:
            trained_model = self.tuner.fit(self.architecture, x, y)

        else:
            trained_model = self.architecture.fit(x, y)
        return Predictor(trained_model)

    def transform_y(self, y):
        # TODO MARK
        return y


class RandomSearchTuner(Tuner):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.hyperparameters = config['hyperparameters']
        self.num_iters = config['num_iters']
        self.num_cv_folds = config['num_cv_folds']
        self.scoring_function = config['scoring_function']
        self.verbose = config['verbose']

    def fit(self, architecture, x, y) -> Predictor:
        random_search = RandomizedSearchCV(estimator=architecture, param_distributions=self.hyperparameters,
                                           n_iter=self.num_iters, cv=self.num_cv_folds, verbose=self.verbose,
                                           random_state=42, n_jobs=-1, scoring=self.scoring_function)
        random_search.fit(x, y)
        best_est = random_search.best_estimator_
        return best_est


class Metric(ABC):

    name = 'Metric'

    def __init__(self, classification_cutoff: float = 0.5):
        self.classification_cutoff = classification_cutoff

    @abstractmethod
    def calculate(self, y_true, y_pred_proba):
        pass

    def convert_proba_to_class(self, y_pred_proba: np.ndarray):
        """
        Convert a probabilty array to a classification based on the classification cutoff. If an array with two columns
         is passed (two class classification), the output is reduced to a single Series.

        Args:
            y_pred_proba: Probabilities for the classification classes.

        Returns: Series of 0s and 1s.
        """
        classification = np.where(y_pred_proba > self.classification_cutoff, 1, 0)
        return classification


class F1_Macro(Metric):

    name = 'f1_macro'

    def calculate(self, y_true, y_pred_proba):
        y_pred = self.convert_proba_to_class(y_pred_proba)
        metric = f1_score(y_true, y_pred, average='macro')
        return metric


class AUPRC(Metric):

    name = 'auprc'

    def calculate(self, y_true, y_pred_proba):
        y_pred = y_pred_proba
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        metric = auc(recall, precision)
        return metric


class AUROC(Metric):

    name = 'auroc'

    def calculate(self, y_true, y_pred_proba):
        y_pred = y_pred_proba
        metric = roc_auc_score(y_true, y_pred)
        return metric


class LogLoss(Metric):

    name = 'log_loss'

    def calculate(self, y_true, y_pred_proba):
        y_pred = y_pred_proba
        metric = log_loss(y_true, y_pred)
        return metric


class Experiment:
    """
    Orchestrate a machine learning experiment, including training and evaluation.
    """

    def __init__(self, data_set: DataSet, stratifier: PartitionedLabelStratifier, trainer: Trainer,
                 metrics: List[Metric]):
        self.data_set = data_set
        self.stratifier = stratifier
        self.stratifier.load_data(self.data_set)
        self.trainer = trainer
        self.metrics = metrics

        self.partition_predictors = []
        self.partition_evaluations = []
        self.evaluation = None

    def go(self):
        for x_train, y_train, x_test, y_test in self.stratifier:
            predictor = self.trainer.fit(x_train, y_train)
            self.partition_predictors.append(predictor)
            partition_evaluation = self.evaluate(predictor, self.metrics, x_test, y_test)
            self.partition_evaluations.append(partition_evaluation)
        self.evaluation = self.summarize_evaluations(self.partition_evaluations)
        return self.evaluation

    def to_dict(self):
        d = {
            'stratifier': self.stratifier.to_dict(),
            'evaluation': self.evaluation.to_dict(),
        }
        return d

    @staticmethod
    def evaluate(predictor: Predictor, metrics: List[Metric], x, y_true) -> Dict[str, Union[int, float]]:
        results = {}
        for metric in metrics:
            y_pred_proba = predictor.predict_proba(x)
            val = metric.calculate(y_true, y_pred_proba)
            results[metric.name] = val
        return results

    @staticmethod
    def summarize_evaluations(partition_evaluations: List[Dict[str, Union[int, float]]]):
        for i, eval_dict in enumerate(partition_evaluations):
            eval_dict['partition'] = i
        evaluation_df = pd.DataFrame(partition_evaluations)
        col_order = ['partition'] + [col for col in evaluation_df.columns if col != 'partition']
        evaluation_df = evaluation_df[col_order]
        return evaluation_df


# === EXAMPLE ===
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
    }
    data_set = DataSet(df, config['DataSet'])
    stratifier = TrainTestStratifier(config['Stratifier'])
    architecture = RandomForestClassifier()
    tuner = RandomSearchTuner(config['Tuner'])
    trainer = Trainer(architecture, config['Trainer'], tuner)
    # trainer = RandomForestClassifier()

    metrics = [F1_Macro(classification_cutoff=0.5), AUPRC(), AUROC(), LogLoss()]

    exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics)
    results = exp.go()
    print(results)


ex()
