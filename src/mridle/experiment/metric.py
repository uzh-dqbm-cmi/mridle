from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import log_loss, f1_score, precision_recall_curve, auc, roc_auc_score
from typing import Dict


class Metric(ABC):

    name = 'Metric'

    def __init__(self, config: Dict = None):
        self.config = config
        if self.config is None:
            self.config = {}

        self.classification_cutoff = self.config.get('classification_cutoff', 0.5)

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
