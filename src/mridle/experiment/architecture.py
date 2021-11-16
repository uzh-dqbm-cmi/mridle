from abc import abstractmethod
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from typing import Dict
import xgboost as xgb


class Architecture(ConfigurableComponent):

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def fit(self, x, y):
        pass


class ArchitectureInterface(ComponentInterface):

    registered_flavors = {
        'RandomForestClassifier': RandomForestClassifier,  # TODO enable auto-loading from sklearn
        'LogisticRegression': LogisticRegression,
        'XGBClassifier': xgb.XGBClassifier,
        'Pipeline': Pipeline
    }

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
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(**d['config'])
        return flavor_instance

    @classmethod
    def deserialize(cls, d: Dict) -> ConfigurableComponent:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        kwargs = cls.additional_info_for_deserialization(d)
        flavor_instance = flavor_cls(**d['config'], **kwargs)
        return flavor_instance

    @classmethod
    def serialize(cls, component) -> Dict:
        if isinstance(component, BaseEstimator):
            return cls.sklearn_to_dict(component)
        else:
            return super().serialize(component)

    @staticmethod
    def sklearn_to_dict(estimator) -> Dict:
        d = {
            'flavor': type(estimator).__name__,
            'config': estimator.get_params(),
        }
        return d
