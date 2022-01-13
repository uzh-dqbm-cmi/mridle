import pandas as pd
from typing import Any, Dict
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .stratifier import Stratifier, StratifierInterface


class DataSet(ConfigurableComponent):
    """
    A dataset class config dictionary contains the following configurable elements:
    - features: name of the features to be used in building the features tensor
    - target: name to be used in building the target tensor
    """

    def __init__(self, config: Dict, data: pd.DataFrame, stratifier: Stratifier):
        if not isinstance(config, dict) or not isinstance(stratifier, Stratifier):
            raise ValueError('It looks like the arguments to `DataSet` are out of order. Please use'
                             ' `DataSet(config, data, stratifier)`.')
        super().__init__(config)
        self.validate_config(config, data)
        self.data = data
        self.features_list = config['features']
        self.target = config['target']
        self.stratifier = stratifier
        if stratifier.data_set is None:
            # stratifier may already contain data if it's being deserialized, in which case we don't want to overwrite
            # the partitions indexes with new ones
            self.stratifier.load_data(self.data)

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

    serialization_schema = {
        'data': {
            'required': False,
        }
    }

    @classmethod
    def additional_info_for_serialization(cls, dataset: DataSet) -> Dict[str, Any]:
        data = dataset.data.to_dict()  # TODO: I feel like I've had problems with this before
        return {'data': data}

    # @classmethod
    # def additional_info_for_deserialization(cls, d: Dict) -> Dict[str, Any]:
    #     df = pd.DataFrame(d['data'])
    #     return {'data': df}

    @classmethod
    def deserialize(cls, d: Dict[str, Dict]) -> DataSet:
        """
        Instantiate a Trainer from dictionary containing keys ['DataSet', 'Stratifier'].

        Args:
            d: A dictionary with keys ['DataSet', 'Stratifier'], each containing configuration dictionaries.
             The configuration dictionaries contain the key 'flavor' describing the class name of the component to be
             instantiated, and key 'config' containing the object's config dictionary. The configuration dictionaries
             may also contain other keys, which must be added to the object by the subclass-ed deserialize method.

        Returns:
            A deserialized Trainer object.
        """
        data_set_config = d['DataSet']
        data_set_config = cls.validate_serialization_config(data_set_config)
        flavor_cls = cls.select_flavor(data_set_config['flavor'])

        df = pd.DataFrame(d['DataSet']['data'])

        stratifier = StratifierInterface.deserialize(d['Stratifier'])

        flavor_instance = flavor_cls(config=data_set_config['config'], data=df, stratifier=stratifier)
        return flavor_instance
