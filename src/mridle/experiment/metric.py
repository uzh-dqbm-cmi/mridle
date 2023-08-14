from abc import abstractmethod
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_recall_curve, auc, roc_auc_score, brier_score_loss, \
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from typing import Dict, List


class Metric(ConfigurableComponent):
    name = 'Metric'
    metric_type = 'Metric_Type'

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.classification_cutoff = self.config.get('classification_cutoff', 0.5)

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
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
    metric_type = 'classification'

    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred = self.convert_proba_to_class(y_pred_proba)
        metric = f1_score(y_true, y_pred, average='macro')
        return metric


class BrierScore(Metric):
    name = 'brier_score'
    metric_type = 'classification'

    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred = y_pred_proba
        metric = brier_score_loss(y_true, y_pred)
        return metric


class AUPRC(Metric):
    name = 'auprc'
    metric_type = 'classification'

    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred = y_pred_proba
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        metric = auc(recall, precision)
        return metric


class AUROC(Metric):
    name = 'auroc'
    metric_type = 'classification'

    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred = y_pred_proba
        metric = roc_auc_score(y_true, y_pred)
        return metric


class LogLoss(Metric):
    name = 'log_loss'
    metric_type = 'classification'

    def calculate(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        y_pred = y_pred_proba
        metric = log_loss(y_true, y_pred)
        return metric


class MSE(Metric):
    name = 'mse'
    metric_type = 'regression'

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = mean_squared_error(y_true, y_pred)
        return metric


class RedFlag(Metric):
    name = 'redflag'
    metric_type = 'regression'

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        abs_error = np.abs(y_true - y_pred)
        pct_error = abs_error/y_true
        metric = np.mean(pct_error > 0.2)
        return metric


class RMSE(Metric):
    name = 'rmse'
    metric_type = 'regression'

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = mean_squared_error(y_true, y_pred, squared=False)
        return metric


class MAE(Metric):
    name = 'mae'
    metric_type = 'regression'

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = mean_absolute_error(y_true, y_pred)
        return metric


class MedianAbsoluteError(Metric):
    name = 'median_absolute_error'
    metric_type = 'regression'

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = median_absolute_error(y_true, y_pred)
        return metric


class MAPE(Metric):
    name = 'mape'
    metric_type = 'regression'

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        metric = mean_absolute_percentage_error(y_true, y_pred)
        return metric


class MetricInterface(ComponentInterface):
    registered_flavors = {
        'F1_Macro': F1_Macro,
        'BrierScore': BrierScore,
        'AUPRC': AUPRC,
        'AUROC': AUROC,
        'LogLoss': LogLoss,
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MedianAbsoluteError': MedianAbsoluteError
    }

    @classmethod
    def serialize(cls, component: List[Metric]) -> List[Dict]:
        list_of_dicts = []
        for metric in component:
            m_dict = super().serialize(metric)
            list_of_dicts.append(m_dict)
        return list_of_dicts

    @classmethod
    def configure(cls, components: List[Dict], **kwargs) -> List[Metric]:
        metrics = []
        for m in components:
            metric = super().configure(m, **kwargs)
            metrics.append(metric)
        return metrics

    @classmethod
    def deserialize(cls, components: List[Dict]) -> List[Metric]:
        metrics = []
        for m in components:
            metric = super().deserialize(m)
            metrics.append(metric)
        return metrics
