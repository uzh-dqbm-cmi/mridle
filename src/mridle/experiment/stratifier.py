from abc import abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict, List, Tuple
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from .dataset import DataSet


class Stratifier(ConfigurableComponent):
    """
    Yield data partitions.
    """

    def __init__(self, config: Dict, partition_idxs: List[Tuple[List[int], List[int]]] = None):
        super().__init__(config)
        self.validate_config(config)
        self.partition_idxs = partition_idxs

    def stratify(self, data_set: DataSet):
        self.partition_idxs = self.partition_data(data_set)

    @property
    def is_stratified(self):
        return self.partition_idxs is not None

    @property
    def n_partitions(self):
        return len(self.partition_idxs)

    @abstractmethod
    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        pass

    @classmethod
    @abstractmethod
    def validate_config(cls, config):
        pass

    def materialize_partition(self, partition_id: int, data_set: DataSet) -> Tuple[pd.DataFrame, pd.Series,
                                                                                   pd.DataFrame, pd.Series]:
        """
        Create training and testing dataset based on the partition, which indicate the ids for the test set.

        Args:
            partition_id: Index of the partition within self.partition_idxs.

        Returns: X_train, y_train, X_test, y_test
        """
        train_partition_ids, test_partition_ids = self.partition_idxs[partition_id]
        x_train = data_set.x.iloc[train_partition_ids]
        y_train = data_set.y.iloc[train_partition_ids]
        x_test = data_set.x.iloc[test_partition_ids]
        y_test = data_set.y.iloc[test_partition_ids]
        return x_train, y_train, x_test, y_test


class PartitionedLabelStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the data_set into n_partitions roughly equal lists, stratified by label."""
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

    def __init__(self, config: Dict, partition_idxs=None):
        super().__init__(config, partition_idxs)
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


class PartitionedFeatureStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Split dataset by feature values of provided column."""
        data_set_copy = data_set.data.copy()
        data_set_copy = data_set_copy.reset_index()
        label_list = data_set_copy[self.config['split_feature']].unique()
        partitions = []
        for l_id, f_label in enumerate(label_list):
            print(f_label)
            train_ids = np.array(data_set_copy[data_set_copy[self.config['split_feature']] != f_label].index)
            test_ids = np.array(data_set_copy[data_set_copy[self.config['split_feature']] == f_label].index)
            partitions.append([train_ids, test_ids])
        return partitions

    @classmethod
    def validate_config(cls, config):
        for key in ['split_feature', ]:
            if key not in config:
                raise ValueError(f"{cls.__name__} config must contain entry '{key}'.")
        return True


class TimeSeriesStratifier(Stratifier):

    def partition_data(self, data_set: DataSet) -> List[Tuple[List[int], List[int]]]:
        """Split dataset by feature values of provided column."""
        data_set_copy = data_set.data.copy()
        data_set_copy = data_set_copy.reset_index()
        time_feature = self.config['time_feature']
        ordered_dates = self.config['ordered_dates']
        test_size_time = (pd.to_datetime(ordered_dates[1]) - pd.to_datetime(ordered_dates[0]))
        partitions = []
        for l_id, d in enumerate(ordered_dates):
            print(d)
            cut_off = pd.to_datetime(d)
            train_ids = np.array(data_set_copy[data_set_copy[time_feature] < cut_off].index)
            test_ids = np.array(data_set_copy[(data_set_copy[time_feature] >= cut_off) &
                                              (data_set_copy[time_feature] < (cut_off + test_size_time))].index)
            print(cut_off + test_size_time)
            partitions.append([train_ids, test_ids])
        return partitions

    @classmethod
    def validate_config(cls, config):
        for key in ['time_feature', 'ordered_dates']:
            if key not in config:
                raise ValueError(f"{cls.__name__} config must contain entry '{key}'.")
        return True


class StratifierInterface(ComponentInterface):

    registered_flavors = {
        'PartitionedFeatureStratifier': PartitionedFeatureStratifier,
        'PartitionedLabelStratifier': PartitionedLabelStratifier,
        'TrainTestStratifier': TrainTestStratifier,
        'TimeSeriesStratifier': TimeSeriesStratifier
    }

    serialization_schema = {
        'partition_idxs': {'required': False, }
    }

    @classmethod
    def additional_info_for_deserialization(cls, d: Dict) -> Dict[str, Any]:
        partition_idxs = d['partition_idxs']
        return {'partition_idxs': partition_idxs}

    @classmethod
    def additional_info_for_serialization(cls, stratifier: Stratifier) -> Dict[str, Any]:
        return {'partition_idxs': stratifier.partition_idxs}
