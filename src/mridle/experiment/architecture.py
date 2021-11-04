from abc import ABC, abstractmethod


class Architecture(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def fit(self, x, y):
        pass
