from sklearn.preprocessing import LabelEncoder
from typing import Dict
from .architecture import Architecture, ArchitectureInterface
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .tuner import Tuner, TunerInterface
from .predictor import Predictor


class Trainer(ConfigurableComponent):

    def __init__(self, config: Dict, architecture: Architecture, tuner: Tuner = None):
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

    # @classmethod
    # def to_dict(self, component) -> Dict:
    #     d = super().to_dict(component)
    #     return d

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> Trainer:
        for required_key in ['flavor', 'config']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        architecture = kwargs['architecture']
        tuner = kwargs['tuner']

        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(config=d['config'], architecture=architecture, tuner=tuner)
        return flavor_instance

    @classmethod
    def from_dict(cls, d: Dict) -> Trainer:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        trainer_config = d['Trainer']
        for required_key in ['flavor', 'config']:
            if required_key not in trainer_config:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        flavor_cls = cls.select_flavor(trainer_config['flavor'])
        architecture = ArchitectureInterface.from_dict(d['Architecture'])
        if 'Tuner' in d:
            tuner = TunerInterface.from_dict(d['Tuner'])
        else:
            tuner = None
        flavor_instance = flavor_cls(config=trainer_config['config'], architecture=architecture, tuner=tuner)
        return flavor_instance
