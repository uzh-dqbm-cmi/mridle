import datetime
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
                 metrics: List[Metric]):
        self.dataset = data_set
        self.stratifier = stratifier
        self.stratifier.load_data(self.dataset)
        self.trainer = trainer
        self.metrics = metrics

        self.run_date = None
        self.partition_predictors = []
        self.partition_evaluations = []
        self.evaluation = pd.DataFrame()

    def go(self):
        self.run_date = datetime.datetime.now()
        for x_train, y_train, x_test, y_test in self.stratifier:
            predictor = self.trainer.fit(x_train, y_train)
            self.partition_predictors.append(predictor)
            partition_evaluation = self.evaluate(predictor, self.metrics, x_test, y_test)
            self.partition_evaluations.append(partition_evaluation)
        self.evaluation = self.summarize_evaluations(self.partition_evaluations)
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

    def to_dict(self) -> Dict:
        return ExperimentInterface.to_dict(self)

    @classmethod
    def load_from_dict(cls, d) -> 'Experiment':
        pass


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
        tuner = TunerInterface.configure(config['Tuner'])
        trainer = TrainerInterface.configure(config['Trainer'], architecture=architecture, tuner=tuner)
        metrics = MetricInterface.configure(config['Metrics'])
        exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics)
        return exp

    @classmethod
    def from_dict(cls, config: Dict) -> Experiment:
        """
        Re-instantiate a experiment object from a dictionary.

        Args:
            config:

        Returns:

        """
        components = config['components']
        data_set = DataSetInterface.from_dict(components['DataSet'])
        stratifier = StratifierInterface.from_dict(components['Stratifier'])
        trainer_dict = {key: components[key] for key in components if key in ['Trainer', 'Architecture, Tuner']}
        trainer = TrainerInterface.from_dict(trainer_dict)
        metrics = MetricInterface.from_dict(components['Metrics'])
        exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics)
        # TODO: set metadata
        return exp

    @classmethod
    def to_dict(cls, experiment: Experiment) -> Dict:
        d = {
            'metadata': {
                'name': 'name',
                'rundate': experiment.run_date,
            },
            'components': {
                'DataSet': DataSetInterface.to_dict(experiment.dataset),
                'Stratifier': StratifierInterface.to_dict(experiment.stratifier),
                'Architecture': ArchitectureInterface.to_dict(experiment.trainer.architecture),  # TODO ??
                'Trainer': TrainerInterface.to_dict(experiment.trainer),
                'Tuner': TunerInterface.to_dict(experiment.trainer.tuner),  # TODO ??
                'Metrics': MetricInterface.to_dict(experiment.metrics),
                'Predictors': experiment.partition_predictors,
                # 'Evaluation': self.evaluation.to_dict(),  # TODO
            },
        }
        return d
