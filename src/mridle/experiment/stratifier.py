from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple
from .dataset import DataSet


class Stratifier(ABC):
    """
    Yield data partitions.
    """

    def __init__(self, config: Dict):
        self.validate_config(config)
        self.n_partitions = config['n_partitions']

        self.partition_idxs = config.get('partition_idxs', None)
        if 'data_set' in config:
            data_set_dict = config['data_set']
            self.data_set = DataSet.from_dict(data_set_dict)
        else:
            self.data_set = None

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Iterate over the partitions.

        Returns: X_train, y_train, X_test, y_test
        """
        if self.data_set is None:
            raise ValueError('Stratifier does not have data. Use `Stratifier.load(data_set)` before iterating.')
        if self.n < self.n_partitions:
            return_value = self.materialize_partition(self.n)
            self.n += 1
            return return_value
        else:
            raise StopIteration

    def to_dict(self):
        d = {
            'n_partitions': self.n_partitions,
            'partition_idxs': self.partition_idxs,
            'data_set': self.data_set.to_dict(),
        }
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d)

    def load_data(self, data_set: DataSet):
        self.data_set = data_set
        self.partition_idxs = self.partition_data(self.data_set)

    @abstractmethod
    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        pass

    def materialize_partition(self, partition_id: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create training and testing dataset based on the partition, which indicate the ids for the test set.

        Args:
            partition_id: Index of the partition within self.partition_idxs.

        Returns: X_train, y_train, X_test, y_test
        """
        train_partition_ids, test_partition_ids = self.partition_idxs[partition_id]
        x_train = self.data_set.x.iloc[train_partition_ids]
        y_train = self.data_set.y.iloc[train_partition_ids]
        x_test = self.data_set.x.iloc[test_partition_ids]
        y_test = self.data_set.y.iloc[test_partition_ids]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def validate_config(config):
        for key in ['n_partitions', ]:
            if key not in config:
                raise ValueError(f"PartitionedLabelStratifier config must contain entry '{key}'.")
        return True


class PartitionedLabelStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the doc_list into n_partitions roughly equal lists, stratified by label."""
        label_list = data_set.y
        skf = StratifiedKFold(n_splits=self.n_partitions, random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = skf.split(x, label_list)
        partitions = []
        for p_id, p in enumerate(partition_indexes):
            partitions.append(p)
        return partitions


class TrainTestStratifier(Stratifier):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.test_split_size = config['test_split_size']

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Split data once into train and test sets. Percentage of data in test set supplied as argument."""
        df_len = len(data_set.x.index)
        perm = np.random.permutation(df_len)
        train_end = int((1-self.test_split_size) * df_len)
        train_idx = perm[:train_end]
        test_idx = perm[train_end:]
        partitions = [(train_idx, test_idx)]
        return partitions
