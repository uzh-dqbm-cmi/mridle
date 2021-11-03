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


class PartitionedStratifier:
    """
    Yield data partitions.
    # TODO: create subclasses LabelStratifier, TrainTestSplitStratifier, RandomStratifier
    """

    def __init__(self, config: Dict):
        self.validate_config(config)
        self.n_partitions = config['n_partitions']
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
            'data_set': self.data_set.to_dict()

        }
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d)

    def load_data(self, data_set: DataSet):
        self.data_set = data_set
        self.partition_idxs = self.partition_data_stratified(self.data_set.y, self.n_partitions)

    @classmethod
    def partition_data_stratified(cls, label_list: pd.Series, n_partitions: int) -> \
            List[Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the doc_list into n roughly equal lists, stratified by label."""
        skf = StratifiedKFold(n_splits=n_partitions, random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = skf.split(x, label_list)
        partitions = []
        for p_id, p in enumerate(partition_indexes):
            partitions.append(p)
        return partitions

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
                raise ValueError(f"DataSet config must contain entry '{key}'.")
        return True


class Predictor:

    def __init__(self, model=None):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class Trainer:

    def __init__(self, model):
        self.model = model

    def fit(self, x, y) -> Predictor:
        return self.model.fit(x, y)


class Tuner:

    def __init__(self, config: Dict):
        self.config = config

    def fit(self, trainer: Trainer, x, y) -> Predictor:
        return trainer.fit(x, y)


class Metric:

    name = 'Metric'

    def calculate(self, predictor: Predictor, x, y_true):
        y_pred = predictor.predict(x)
        return y_pred == y_true


class F1_Macro_Metric(Metric):

    name = 'f1_macro'

    def calculate(self, predictor: Predictor, x, y_true):
        y_pred = predictor.predict(x)
        metric = f1_score(y_true, y_pred, average='macro')
        return metric


class AUPRC_Metric(Metric):

    name = 'auprc'

    def calculate(self, predictor: Predictor, x, y_true):
        y_pred = predictor.predict_proba(x)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        metric = auc(recall, precision)
        return metric


class AUROC_Metric(Metric):

    name = 'auroc'

    def calculate(self, predictor: Predictor, x, y_true):
        y_pred = predictor.predict_proba(x)[:, 1]
        metric = roc_auc_score(y_true, y_pred)
        return metric


class LogLoss_Metric(Metric):

    name = 'log_loss'

    def calculate(self, predictor: Predictor, x, y_true):
        y_pred = predictor.predict_proba(x)[:, 1]
        metric = log_loss(y_true, y_pred)
        return metric


class Experiment:
    """
    Orchestrate a machine learning experiment, including training and evaluation.
    """

    def __init__(self, data_set: DataSet, stratifier: PartitionedStratifier, trainer: Trainer, metrics: List[Metric],
                 tuner: Tuner = None):
        self.data_set = data_set
        self.stratifier = stratifier
        self.stratifier.load_data(self.data_set)
        self.trainer = trainer
        self.metrics = metrics
        self.tuner = tuner

        self.partition_predictors = []
        self.partition_evaluations = []
        self.evaluation = None

    def go(self):
        for x_train, y_train, x_test, y_test in self.stratifier:
            if self.tuner:
                predictor = self.tuner.fit(self.trainer, x_train, y_train)
            else:
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

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
df = df.dropna().copy()

data_set_config = {
    'features': [
        'pclass',
        'age',
        'adult_male',
        'sibsp',
        'parch'
    ],
    'target': 'survived',
}
data_set = DataSet(df, data_set_config)

data_set.x

stratifier_config = {
    'n_partitions': 5,
}
stratifier = PartitionedStratifier(stratifier_config)

architecture = RandomForestClassifier()
trainer = Trainer(architecture)
tuner = Tuner({})

metrics = [F1_Macro_Metric(cutoff=0.5), AUPRC_Metric(), AUROC_Metric(), LogLoss_Metric()]

# TODO: just make Tuner a subclass of Trainer, so Experiment only gets 1?
# TODO: They're just objects with a .fit() method as far as Experiment is concerned
exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics, tuner=None)
results = exp.go()
print(results)
