from sklearn.preprocessing import LabelEncoder
from typing import Dict
from .architecture import Architecture, ArchitectureInterface
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .tuner import Tuner, TunerInterface
from .predictor import Predictor


class Trainer(ConfigurableComponent):

    def __init__(self, config: Dict, architecture: Architecture, tuner: Tuner = None):
        if not isinstance(config, dict):
            if isinstance(architecture, dict) or isinstance(tuner, dict):
                raise ValueError('It looks like the order of the arguments to `Trainer` is swapped. Please use'
                                 ' `Trainer(config, architecture, tuner)`.')
            raise ValueError('The first argument to DataSet must be a dictionary.')
        super().__init__(config)
        self.architecture = architecture
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


class TrainerInterface(ComponentInterface):

    registered_flavors = {
        'Trainer': Trainer,
        'SkorchTrainer': SkorchTrainer,
    }

    @classmethod
    def deserialize(cls, d: Dict) -> Trainer:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        trainer_config = d['Trainer']
        trainer_config = cls.validate_serialization_config(trainer_config)

        flavor_cls = cls.select_flavor(trainer_config['flavor'])
        architecture = ArchitectureInterface.deserialize(d['Architecture'])
        if 'Tuner' in d:
            tuner = TunerInterface.deserialize(d['Tuner'])
        else:
            tuner = None
        flavor_instance = flavor_cls(config=trainer_config['config'], architecture=architecture, tuner=tuner)
        return flavor_instance
