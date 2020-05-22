from mridle.mridle.experiment import ModelRun
import pandas as pd
from typing import Any, Dict, List, Tuple, Callable


class HarveyModel(ModelRun):

    @classmethod
    def build_x_features(cls, data_set: Any, encoders: Dict) -> Tuple[pd.DataFrame, List]:
        """
        Build custom features

        Args:
            data_set: Data set to transform into features.
            encoders: Dict of pre-trained encoders for use in building features.

        Returns:
            dataframe
            List of features
        """
        feature_columns = ['historic_no_show_cnt', 'days_sched_in_advance', 'sched_for_hour']
        return data_set[feature_columns].copy(), feature_columns

