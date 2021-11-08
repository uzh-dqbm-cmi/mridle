from abc import abstractmethod
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from typing import Dict


class Architecture(ConfigurableComponent):

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def fit(self, x, y):
        pass


class ArchitectureInterface(ComponentInterface):

    registered_flavors = {
        'RandomForestClassifier': RandomForestClassifier,  # TODO enable auto-loading from sklearn
    }

    # TODO: make not duplicate method!
    @classmethod
    def configure(cls, d: Dict, **kwargs) -> ConfigurableComponent:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        for required_key in ['flavor', 'config']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(**d.get('config', {}))
        return flavor_instance

    @classmethod
    def from_dict(cls, d: Dict) -> ConfigurableComponent:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        for required_key in ['flavor', 'config']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(**d.get('config', {}))
        return flavor_instance

    @classmethod
    def to_dict(cls, component) -> Dict:
        if isinstance(component, BaseEstimator):
            return cls.sklearn_to_dict(component)
        else:
            return super().to_dict(component)

    @staticmethod
    def sklearn_to_dict(estimator) -> Dict:
        d = {
            'flavor': type(estimator).__name__,
            'config': estimator.get_params(),
        }
        return d
