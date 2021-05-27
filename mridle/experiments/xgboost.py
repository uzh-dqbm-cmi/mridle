from mridle.experiment import ModelRun
import pandas as pd
from typing import Any, Dict, List, Tuple

y_column = 'NoShow'


class CustomXGB(ModelRun):

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
        feature_columns = list(data_set.columns)
        feature_columns.remove(y_column)

        return data_set[feature_columns].copy(), feature_columns


def process_features_for_model(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Changes variables for model optimization modifying feature_df

    Args:
       dataframe: dataframe obtained from feature generation

    Returns: modified dataframe specific for this model
    """

    dataframe['sched_2_days'] = dataframe['sched_days_advanced'] <= 2
    dataframe['close_to_usz'] = dataframe['distance_to_usz'] < 16

    return dataframe
