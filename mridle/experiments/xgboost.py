from mridle.experiment import ModelRun
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
import xgboost as xgb

# cols_for_modeling = ['no_show_before', 'no_show_before_sq', 'sched_days_advanced', 'hour_sched',
#                      'distance_to_usz', 'age', 'close_to_usz', 'male', 'female', 'age_sq',
#                      'sched_days_advanced_sq', 'distance_to_usz_sq', 'sched_2_days', 'age_20_60']

# max_depth = [2, 4, 10, 50, 100, None]
# n_estimators = np.arange(10, 800, dtype=int)
# learning_rate = np.arange(np.log(0.005), np.log(0.2))

# hyperparams = {'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate}
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


# def process_features_for_model(dataframe: pd.DataFrame) -> pd.DataFrame:
#   """
#    Changes variables for model optimization modifying feature_df
#
#    Args:
#        dataframe: dataframe obtained from feature generation
#
#    Returns: modified dataframe specific for this model
#    """
#
#    dataframe['sched_2_days'] = dataframe['sched_days_advanced'] <= 2
#    dataframe['close_to_usz'] = dataframe['distance_to_usz'] < 16
#
#    return dataframe