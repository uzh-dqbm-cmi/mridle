from sklearn.preprocessing import LabelEncoder
from typing import Dict
from .architecture import Architecture
from .tuner import Tuner
from .predictor import Predictor


class Trainer:

    def __init__(self, architecture: Architecture, config: Dict, tuner: Tuner = None):
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

    @staticmethod
    def transform_y(y):
        y = LabelEncoder().fit_transform(y)
        y = y.astype('float32')
        y = y.reshape((len(y), 1))
        return y
