from mridle.experiment import ModelRun
import pandas as pd
from typing import Any, Dict, List, Tuple

y_column = 'NoShow'


class CustomXGB(ModelRun):

    @classmethod
    def build_x_features(cls, data_set: Any, encoders: Dict, cat_columns: List) -> Tuple[pd.DataFrame, List]:
        """
        Build custom features

        Args:
            data_set: Data set to transform into features.
            encoders: Dict of pre-trained encoders for use in building features.
            cat_columns: List of categorical columns, for which dummy features will be created.

        Returns:
            dataframe
            List of features
        """

        print(cat_columns)
        xgb_data = pd.get_dummies(data_set, columns=cat_columns)
        cols = list(xgb_data.columns)
        cols.remove(y_column)

        return xgb_data[cols].copy(), cols


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
