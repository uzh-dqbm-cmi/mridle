from abc import abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict, List, Tuple
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .dataset import DataSet, DataSetInterface


class Stratifier(ConfigurableComponent):
    """
    Yield data partitions.
    """

    def __init__(self, config: Dict, data_set: DataSet = None, partition_idxs: List[Tuple[List[int], List[int]]] = None
                 ):
        super().__init__(config)
        self.validate_config(config)

        if data_set is not None and partition_idxs is not None:
            self.partition_idxs = partition_idxs
            self.data_set = data_set
        elif data_set is None and partition_idxs is None:
            self.partition_idxs = None
            self.data_set = None
        else:
            raise ValueError(f"{type(self).__name__} received one but not both of data_set and partition_idxs. It is"
                             f" only valid to pass none or both.")

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

    def load_data(self, data_set: DataSet):
        self.data_set = data_set
        self.partition_idxs = self.partition_data(self.data_set)

    @property
    def n_partitions(self):
        return len(self.partition_idxs)

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


class PartitionedLabelStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the doc_list into n_partitions roughly equal lists, stratified by label."""
        label_list = data_set.y
        skf = StratifiedKFold(n_splits=self.config['n_partitions'], random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = skf.split(x, label_list)
        partitions = []
        for p_id, p in enumerate(partition_indexes):
            partitions.append(p)
        return partitions

    @classmethod
    def validate_config(cls, config):
        for key in ['n_partitions', ]:
            if key not in config:
                raise ValueError(f"{cls.__name__} config must contain entry '{key}'.")
        return True


class TrainTestStratifier(Stratifier):

    def __init__(self, config: Dict, data_set=None, partition_idxs=None):
        super().__init__(config, data_set, partition_idxs)
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

    @classmethod
    def validate_config(cls, config):
        for key in ['test_split_size', ]:
            if key not in config:
                raise ValueError(f"{cls.__name__} config must contain entry '{key}'.")
        return True


class StratifierInterface(ComponentInterface):

    registered_flavors = {
        'PartitionedLabelStratifier': PartitionedLabelStratifier,
        'TrainTestStratifier': TrainTestStratifier,
    }

    serialization_schema = {
        'partition_idxs': {'required': False, }
    }

    @classmethod
    def deserialize(cls, d: Dict) -> Stratifier:
        stratifier_config = d['Stratifier']
        stratifier_config = cls.validate_serialization_config(stratifier_config)

        flavor_cls = cls.select_flavor(stratifier_config['flavor'])
        data_set = DataSetInterface.deserialize(d['DataSet'])
        partition_idxs = stratifier_config['partition_idxs']
        flavor_instance = flavor_cls(config=stratifier_config['config'], data_set=data_set,
                                     partition_idxs=partition_idxs)
        return flavor_instance

    @classmethod
    def additional_info_for_serialization(cls, stratifier: Stratifier) -> Dict[str, Any]:
        return {'partition_idxs': stratifier.partition_idxs}
