import pandas as pd
from .dataset import DataSet
from .stratifier import Stratifier, PartitionedLabelStratifier, TrainTestStratifier
from .architecture import Architecture
from .trainer import Trainer, SkorchTrainer
from .tuner import Tuner, RandomSearchTuner, BayesianTuner
from .predictor import Predictor
from .metric import Metric, F1_Macro, AUPRC, AUROC, LogLoss
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Union, Type


ComponentType = Union[Type[DataSet], Type[Stratifier], Type[Architecture], Type[Trainer], Type[Tuner], Type[Metric]]


class Experiment:
    """
    Orchestrate a machine learning experiment, including training and evaluation.
    """

    def __init__(self, data_set: DataSet, stratifier: Stratifier, trainer: Trainer,
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

    @classmethod
    def configure(cls, config: Dict, data: pd.DataFrame) -> 'Experiment':
        return ExperimentConfigurator.configure(config=config, data=data)


class ExperimentConfigurator:

    component_flavors = {
        'DataSet': {
            'DataSet': DataSet,
        },
        'Stratifier': {
            'PartitionedLabelStratifier': PartitionedLabelStratifier,
            'TrainTestStratifier': TrainTestStratifier,
        },
        'Architecture': {
            'RandomForestClassifier': RandomForestClassifier,
        },
        'Trainer': {
            'Trainer': Trainer,
            'SkorchTrainer': SkorchTrainer,
        },
        'Tuner': {
            'RandomSearchTuner': RandomSearchTuner,
            'BayesianTuner': BayesianTuner,
        },
        'Metric': {
            'F1_Macro': F1_Macro,
            'AUPRC': AUPRC,
            'AUROC': AUROC,
            'LogLoss': LogLoss,
        }
    }

    @classmethod
    def configure(cls, config: Dict, data: pd.DataFrame) -> Experiment:
        data_set = cls.configure_data_set(data=data, config=config['DataSet'])
        stratifier = cls.configure_stratifier(config['Stratifier'])
        trainer = cls.configure_trainer(config['Architecture'], config['Trainer'], config.get('Tuner', None))
        metrics = cls.configure_metrics(config['Metric'])
        exp = Experiment(data_set=data_set, stratifier=stratifier, trainer=trainer, metrics=metrics)
        return exp

    @classmethod
    def configure_data_set(cls, data: pd.DataFrame, config: Dict) -> DataSet:
        data_set_cls = cls.select_data_set(config['flavor'])
        data_set = data_set_cls(data=data, config=config['config'])
        return data_set

    @classmethod
    def configure_stratifier(cls, config: Dict) -> Stratifier:
        stratifier_cls = cls.select_stratifier(config['flavor'])
        stratifier = stratifier_cls(config=config['config'])
        return stratifier

    @classmethod
    def configure_trainer(cls, architecture_config: Dict, trainer_config: Dict, tuner_config: Dict = None) -> Trainer:
        architecture_cls = cls.select_architecture(architecture_config['flavor'])
        # TODO does this expansion work?
        architecture = architecture_cls(**architecture_config.get('config', {}))

        if tuner_config:
            tuner_cls = cls.select_tuner(tuner_config['flavor'])
            tuner = tuner_cls(config=tuner_config.get('config', None))
        else:
            tuner = None

        trainer_cls = cls.select_trainer(trainer_config['flavor'])
        trainer = trainer_cls(architecture=architecture, config=trainer_config['config'], tuner=tuner)
        return trainer

    @classmethod
    def configure_metrics(cls, config: List[Dict]) -> List[Metric]:
        metrics = []
        for m in config:
            metric_cls = cls.select_metric(m['flavor'])
            metric = metric_cls(config=m.get('config', None))
            metrics.append(metric)
        return metrics

    @classmethod
    def select_component(cls, flavor: str, component_type: str) -> ComponentType:
        component_flavors = cls.component_flavors[component_type]
        if flavor in component_flavors:
            return component_flavors[flavor]
        else:
            raise ValueError(f"{component_type} '{flavor}' not recognized")

    @classmethod
    def select_data_set(cls, flavor) -> Type[DataSet]:
        return cls.select_component(flavor, component_type='DataSet')

    @classmethod
    def select_stratifier(cls, flavor) -> Type[Stratifier]:
        return cls.select_component(flavor, component_type='Stratifier')

    @classmethod
    def select_architecture(cls, flavor) -> Type[Architecture]:
        return cls.select_component(flavor, component_type='Architecture')

    @classmethod
    def select_trainer(cls, flavor) -> Type[Trainer]:
        return cls.select_component(flavor, component_type='Trainer')

    @classmethod
    def select_tuner(cls, flavor) -> Type[Tuner]:
        return cls.select_component(flavor, component_type='Tuner')

    @classmethod
    def select_metric(cls, flavor) -> Type[Metric]:
        return cls.select_component(flavor, component_type='Metric')
