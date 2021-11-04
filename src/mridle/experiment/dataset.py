import pandas as pd
from typing import Dict


class DataSet:
    """
    A dataset class config dictionary contains the following configurable elements:
    - features: name of the features to be used in building the features tensor
    - targets: name of the features to be used in building the targets tensor
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
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

    def to_dict(self):
        d = {
            'data': self.data.to_dict(),  # TODO: I feel like I've had problems with this before
            'features': self.features_list,
            'target': self.target,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        data = pd.DataFrame(d['data'])
        config = d
        return cls(data, config)

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
