import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold  # noqa
from sklearn.metrics import brier_score_loss, log_loss, f1_score  # noqa
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Tuple, Union


class DataSet:
    """
    A dataset class config dictionary contains the following configurable elements:
    - features: name of the features to be used in building the features tensor
    - targets: name of the features to be used in building the targets tensor
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
        self._validate(config, data)
        self.data = data
        self.features_list = config['features']
        self.target = config['target']

    @property
    def x(self) -> pd.DataFrame:
        return self.data[self.features_list]

    @property
    def y(self) -> pd.Series:
        return self.data[self.target]

    def _validate(self, config, data):
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
        self._validate(config)
        self.n_partitions = config['n_partitions']
        self.partition_idxs = None
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

    def _validate(self, config):
        for key in ['n_partitions', ]:
            if key not in config:
                raise ValueError(f"DataSet config must contain entry '{key}'.")
        return True


class Model:

    def __init__(self, model=None):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class ModelTrainer:

    def __init__(self, model):
        self.model = model

    def fit(self, x_train, y_train) -> Model:
        return self.model.fit(x_train, y_train)


class Tuner:

    def __init__(self, config: Dict):
        self.config = config

    def fit(self, model_trainer: ModelTrainer, x_train, y_train) -> Model:
        return model_trainer.fit(x_train, y_train)


class Metric:

    name = 'Metric'

    def calculate(self, model: Model, x, y_true):
        y_pred = model_trainer.predict(x)
        return y_pred == y_true


class F1_Macro_Metric(Metric):

    name = 'f1_macro'

    def calculate(self, model: Model, x, y_true):
        y_pred = model.predict(x)
        metric = f1_score(y_true, y_pred, average='macro')
        return metric


class ExperimentOrchestrator:
    """
    Orchestrate a machine learning experiment, including training and evaluation.
    """

    def __init__(self, data_set: DataSet, stratifier: PartitionedStratifier, model_trainer: ModelTrainer,
                 metrics: List[Metric], tuner: Tuner = None):
        self.data_set = data_set
        self.stratifier = stratifier
        self.stratifier.load_data(self.data_set)
        self.model_trainer = model_trainer
        self.metrics = metrics
        self.tuner = tuner

        self.partition_models = []
        self.partition_evaluations = []
        self.evaluation = None

    def go(self):
        for x_train, y_train, x_test, y_test in self.stratifier:
            if self.tuner:
                model = self.tuner.fit(self.model_trainer, x_train, y_train)
            else:
                model = self.model_trainer.fit(x_train, y_train)
            self.partition_models.append(model)
            partition_evaluation = evaluate(model, self.metrics, x_test, y_test)
            self.partition_evaluations.append(partition_evaluation)
        self.evaluation = summarize_evaluations(self.partition_evaluations)
        return self.evaluation


def evaluate(model: Model, metrics: List[Metric], x, y) -> Dict[str, Union[int, float]]:
    results = {}
    for metric in metrics:
        val = metric.calculate(model, x, y)
        results[metric.name] = val
    return results


def summarize_evaluations(partition_evaluations: List[Dict[str, Union[int, float]]]):
    for i, eval_dict in enumerate(partition_evaluations):
        eval_dict['partition'] = i
    evaluation = pd.DataFrame(partition_evaluations)
    return evaluation


# === EXAMPLE ===

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

data_set_config = {
    'features': [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
    ],
    'target': 'species',
}
data_set = DataSet(df, data_set_config)

stratifier_config = {
    'n_partitions': 5,
}
stratifier = PartitionedStratifier(stratifier_config)

model = RandomForestClassifier()
model_trainer = ModelTrainer(model)
tuner = Tuner({})

metrics = [F1_Macro_Metric()]

exp = ExperimentOrchestrator(data_set=data_set, stratifier=stratifier, model_trainer=model_trainer, metrics=metrics,
                             tuner=None)
results = exp.go()
print(results)
