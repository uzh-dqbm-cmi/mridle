from datetime import datetime
import pandas as pd
from .dataset import DataSet, DataSetInterface
from .stratifier import Stratifier, StratifierInterface
from .StratifiedDataSet import StratifiedDataSet, StratifiedDataSetInterface
from .architecture import Architecture, ArchitectureInterface
from .trainer import Trainer, TrainerInterface
from .tuner import Tuner, TunerInterface
from .predictor import Predictor
from .metric import Metric, MetricInterface
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Union, Type


ComponentType = Union[Type[DataSet], Type[Stratifier], Type[Architecture], Type[Trainer], Type[Tuner], Type[Metric]]


class Experiment:
    """
    Orchestrate a machine learning experiment, including training and evaluation.

    Experiment takes a DataSet, Stratifier, Trainer (which includes an Architecture and possibly also a Tuner), and a
     list of Metrics, and then executes a complete model training and evaluation with a single `go` call.

     After calling `go`, `Experiment` will contain result in the following fields:
     * `partition_predictors`: A list of `Predictor` objects generated by each partition.
     * `partition_evaluations`: A list of dictionaries containing evaluation metrics for each partition.
     * `partition_training_metadata`: A list of dictionaries that contain metadata about each partition's training
         process. `Trainer`s and `Tuner`s may save information here.
     * `evaluation`: A dataframe where each row is a partition, and columns contain metric values.
     * `final_predictor`: A Predictor trained on the full dataset (not just a partition).
     * `final_training_metadata`: A dictionary of metadata about the training of the `final_predictor`, as set by
        `Trainer` or `Tuner`.
    """

    def __init__(self, data_set: DataSet, stratifier: Stratifier, trainer: Trainer, metrics: List[Metric],
                 metadata: Dict = None):
        """
        Instantiate an `Experiment`.

        Args:
            data_set: A `DataSet`. Will be converted into a `StratifiedDataSet` by `Experiment`.
            stratifier: A `Stratifier`. Will be combined with `DataSet` into a `StratifiedDataSet` by `Experiment`.
            trainer: A `Trainer`. Must already be instatiated with an `Architecture`, and possible also a `Tuner`.
            metrics: A list of `Metric`s to evaluate the trained models on.
            metadata: A dictionary containing any metadata the user wishes to specify about the experiment.
        """
        self.stratified_dataset = StratifiedDataSet(data_set, stratifier)
        self.trainer = trainer
        self.metrics = metrics

        self.metadata = metadata if metadata else dict()

        # results
        self.partition_predictors = []
        self.partition_evaluations_train = []
        self.partition_evaluations_test = []
        self.partition_training_metadata = []
        self.evaluation = pd.DataFrame()

        self.final_predictor = None
        self.final_training_metadata = {}

    def go(self):
        """
        Run the `Experiment`.

        First, save the timestamp of the runtime in the `metadata` dictionary. Then, iterate through each partition
         generated by the `StratifiedDataSet` and run the `Trainer`. Save the resulting `Predictor` in
         `partition_predictors`. Evaluate the `Predictor` on the `Metric`s. Compile the metric evaluations of all
         partitions in the `evaluation` dataframe. Finally, run `Trainer` on the full dataset, and save the result in
         `final_predictor`.

        Returns: A dataframe where each row is a partition, and columns contain metric values.
        """
        self.metadata['run_date'] = datetime.now()
        for i, (x_train, y_train, x_test, y_test) in enumerate(self.stratified_dataset):
            print('Running partition {}...'.format(i+1))
            predictor, training_metadata = self.trainer.fit(x_train, y_train)
            self.partition_predictors.append(predictor)
            self.partition_training_metadata.append(training_metadata)
            partition_evaluation_train = self.evaluate(predictor, self.metrics, x_train, y_train)
            self.partition_evaluations_train.append(partition_evaluation_train)
            partition_evaluation_test = self.evaluate(predictor, self.metrics, x_test, y_test)
            self.partition_evaluations_test.append(partition_evaluation_test)
        self.evaluation_train = self.summarize_evaluations(self.partition_evaluation_train)
        self.evaluation_test = self.summarize_evaluations(self.partition_evaluation_test)

        print('Fitting final model...')
        self.final_predictor, self.final_training_metadata = self.trainer.fit(self.stratified_dataset.x,
                                                                              self.stratified_dataset.y)
        return print("Test Partition Results: ", self.evaluation_test)

    @staticmethod
    def evaluate(predictor: Predictor, metrics: List[Metric], x, y_true) -> Dict[str, Union[int, float]]:
        """
        Given a Predictor, a list of Metrics, and an x and y, compile a dictionary of evaluation results in the form
        {metric_name: metric value}.

        Args:
            predictor: A Predictor object to evaluate.
            metrics: List of metrics to evaluate the Predictor on.
            x: A input dataset for the Predictor.
            y_true: The true labels for the x dataset.

        Returns: a dictionary of evaluation results in the form {metric_name: metric value}.

        """
        results = {}
        for metric in metrics:
            y_pred_proba = predictor.predict_proba(x)
            val = metric.calculate(y_true, y_pred_proba)
            results[metric.name] = val
        return results

    @staticmethod
    def summarize_evaluations(partition_evaluations: List[Dict[str, Union[int, float]]]):
        """
        Compile a list of evaluation dictionaries into a single dataframe.
        Args:
            partition_evaluations: a list of evaluation dictionaries (generated by `Experiment.evaluate(...)`

        Returns: A dataframe where each row is a partition, and columns contain metric values.

        """
        for i, eval_dict in enumerate(partition_evaluations):
            eval_dict['partition'] = i
        evaluation_df = pd.DataFrame(partition_evaluations)
        col_order = ['partition'] + [col for col in evaluation_df.columns if col != 'partition']
        evaluation_df = evaluation_df[col_order]
        return evaluation_df

    @classmethod
    def configure(cls, config: Dict, data: pd.DataFrame) -> 'Experiment':
        return ExperimentInterface.configure(config=config, data=data)

    def serialize(self) -> Dict:
        return ExperimentInterface.serialize(self)

    @classmethod
    def deserialize(cls, d: Dict) -> 'Experiment':
        return ExperimentInterface.deserialize(d)


class ExperimentInterface:

    component_flavors = {
        'Architecture': {
            'RandomForestClassifier': RandomForestClassifier,
        },
    }

    @classmethod
    def configure(cls, config: Dict, data: pd.DataFrame) -> Experiment:
        """
        Instantiate a new Experiment object from a config and dataframe.
        Args:
            config:
            data:

        Returns:

        """
        data_set = DataSetInterface.configure(config['DataSet'], data=data)
        stratifier = StratifierInterface.configure(config['Stratifier'])

        architecture = ArchitectureInterface.configure(config['Architecture'])
        tuner = TunerInterface.configure(config['Tuner']) if 'Tuner' in config else None
        trainer = TrainerInterface.configure(config['Trainer'], architecture=architecture, tuner=tuner)

        metrics = MetricInterface.configure(config['Metrics'])
        metadata = config.get('metadata', dict())

        exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics, metadata=metadata)
        return exp

    @classmethod
    def deserialize(cls, config: Dict) -> Experiment:
        """
        Re-instantiate a experiment object from a dictionary.

        Args:
            config:

        Returns:

        """
        components = config['components']

        data_set = DataSetInterface.deserialize(components['DataSet'])
        stratifier = StratifierInterface.deserialize(components['Stratifier'])

        trainer_dict = {key: components[key] for key in components if key in ['Trainer', 'Architecture', 'Tuner']}
        trainer = TrainerInterface.deserialize(trainer_dict)

        metrics = MetricInterface.deserialize(components['Metrics'])
        exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics,
                         metadata=config['metadata'])

        exp.partition_predictors = config['results']['partition_predictors']
        exp.evaluation = pd.DataFrame(config['results']['evaluation'])
        exp.final_predictor = config['results']['final_predictor']
        exp.partition_training_metadata = config['results'].get('partition_training_metadata', list())  # backwards com.
        exp.final_training_metadata = config['results'].get('final_training_metadata', dict())  # backwards compatibili.
        return exp

    @classmethod
    def serialize(cls, experiment: Experiment) -> Dict:
        d = {
            'metadata': experiment.metadata,
            'components': {
                'DataSet': StratifiedDataSetInterface.serialize(experiment.stratified_dataset),
                'Stratifier': StratifierInterface.serialize(experiment.stratified_dataset.stratifier),
                'Architecture': ArchitectureInterface.serialize(experiment.trainer.architecture),
                'Trainer': TrainerInterface.serialize(experiment.trainer),
                'Metrics': MetricInterface.serialize(experiment.metrics),
            },
            'results': {
                'partition_predictors': experiment.partition_predictors,
                'evaluation': experiment.evaluation.to_dict(),
                'final_predictor': experiment.final_predictor,
                'partition_training_metadata': experiment.partition_training_metadata,
                'final_training_metadata': experiment.final_training_metadata,
            }
        }
        # optional components
        if experiment.trainer.tuner:
            d['components']['Tuner'] = TunerInterface.serialize(experiment.trainer.tuner)
        return d
