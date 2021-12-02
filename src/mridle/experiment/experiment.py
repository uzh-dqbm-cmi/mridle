from datetime import datetime
import pandas as pd
from .dataset import DataSet, DataSetInterface
from .stratifier import Stratifier, StratifierInterface
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
    """

    def __init__(self, data_set: DataSet, stratifier: Stratifier, trainer: Trainer,
                 metrics: List[Metric], metadata: Dict = None):
        self.dataset = data_set
        self.stratifier = stratifier
        if stratifier.data_set is None:
            self.stratifier.load_data(self.dataset)
        self.trainer = trainer
        self.metrics = metrics

        self.metadata = metadata if metadata else dict()

        # results
        self.partition_predictors = []
        self.partition_evaluations = []
        self.evaluation = pd.DataFrame()
        self.final_predictor = None
        self.partition_training_metadata = []
        self.final_training_metadata = {}

    def go(self):
        self.metadata['run_date'] = datetime.now()
        for i, (x_train, y_train, x_test, y_test) in enumerate(self.stratifier):
            print('Running partition {}...'.format(i+1))
            predictor, training_metadata = self.trainer.fit(x_train, y_train)
            self.partition_predictors.append(predictor)
            self.partition_training_metadata.append(training_metadata)
            partition_evaluation = self.evaluate(predictor, self.metrics, x_test, y_test)
            self.partition_evaluations.append(partition_evaluation)
        self.evaluation = self.summarize_evaluations(self.partition_evaluations)

        print('Fitting final model...')
        self.final_predictor, self.final_training_metadata = self.trainer.fit(self.dataset.x, self.dataset.y)
        return self.evaluation

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

    @classmethod
    def configure(cls, config: Dict, data: pd.DataFrame) -> 'Experiment':
        return ExperimentInterface.configure(config=config, data=data)

    def serialize(self) -> Dict:
        return ExperimentInterface.serialize(self)

    @classmethod
    def deserialize(cls, d) -> 'Experiment':
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
        stratifier_dict = {key: components[key] for key in components if key in ['Stratifier', 'DataSet']}
        stratifier = StratifierInterface.deserialize(stratifier_dict)
        trainer_dict = {key: components[key] for key in components if key in ['Trainer', 'Architecture', 'Tuner']}
        trainer = TrainerInterface.deserialize(trainer_dict)
        metrics = MetricInterface.deserialize(components['Metrics'])
        exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics)

        exp.metadata = config['metadata']

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
                'DataSet': DataSetInterface.serialize(experiment.dataset),
                'Stratifier': StratifierInterface.serialize(experiment.stratifier),
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
