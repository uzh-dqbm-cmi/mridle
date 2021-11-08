import pandas as pd
from typing import Dict
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface


class DataSet(ConfigurableComponent):
    """
    A dataset class config dictionary contains the following configurable elements:
    - features: name of the features to be used in building the features tensor
    - targets: name of the features to be used in building the targets tensor
    """

    registered_flavors = {

    }

    def __init__(self, config: Dict, data: pd.DataFrame):
        super().__init__(config)
        self.validate_config(config, data)
        self.data = data
        self.features_list = config['features']
        self.target = config['target']

    @property
    def x(self) -> pd.DataFrame:
        return self.data[self.features_list]

    @property
    def y(self) -> pd.Series:
        return self.data[self.target]

    @staticmethod
    def validate_config(config, data):
        """Make sure the config aligns with the data (referenced columns exist)."""
        for key in ['features', 'target']:
            if key not in config:
                raise ValueError(f"DataSet config must contain entry '{key}'.")

        for col in config['features']:
            if col not in data.columns:
                raise ValueError(f'Feature column {col} not found in dataset.')

        if config['target'] not in data.columns:
            raise ValueError(f"Target column {config['target']} not found in dataset.")

        return True


class DataSetInterface(ComponentInterface):

    registered_flavors = {
        'DataSet': DataSet,
    }

    @classmethod
    def to_dict(self, component) -> Dict:
        d = super().to_dict(component)
        d['data'] = component.data.to_dict(),  # TODO: I feel like I've had problems with this before
        return d

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> DataSet:
        for required_key in ['flavor', 'config']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        data = kwargs['data']

        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(config=d['config'], data=data)
        return flavor_instance

    @classmethod
    def from_dict(cls, d: Dict) -> DataSet:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        for required_key in ['flavor', 'config', 'data']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        flavor_cls = cls.select_flavor(d['flavor'])
        data = pd.DataFrame(d['data'])
        flavor_instance = flavor_cls(config=d['config'], data=data)
        return flavor_instance
