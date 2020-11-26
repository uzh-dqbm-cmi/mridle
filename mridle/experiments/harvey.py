from mridle.experiment import ModelRun
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Tuple


class HarveyModel(ModelRun):

    @classmethod
    def train_encoders(cls, train_set: Any) -> Dict[str, Any]:
        """
        Placeholder function to hold the custom encoder training functionality of a ModelRun.
        By default, returns an empty dictionary.

        Args:
            train_set: Data set to train encoders on.

        Returns:
            Dict of encoders.
        """
        autoscaler = StandardScaler()
        features = ['historic_no_show_cnt', 'no_show_before']
        train_set = train_set.copy()
        train_set[features] = autoscaler.fit_transform(train_set[features])

        return {
            'autoscaler': autoscaler
        }

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
