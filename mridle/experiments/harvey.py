from mridle.experiment import ModelRun
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Tuple

cols_for_modeling = ['no_show_before', 'no_show_before_sq', 'sched_days_advanced', 'hour_sched',
                     'distance_to_usz', 'age', 'close_to_usz', 'male', 'female', 'age_sq',
                     'sched_days_advanced_sq', 'distance_to_usz_sq', 'sched_2_days', 'age_20_60']


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
        feature_columns = cols_for_modeling
        return data_set[feature_columns].copy(), feature_columns

    @classmethod
    def get_test_data_set(cls):
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(cols_for_modeling))), columns=cols_for_modeling)
        df['noshow'] = np.where(df[cols_for_modeling[0]] > 50, 1, 0)
        return df


def process_features_for_model(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes variables for model optimization modifying feature_df

    Args:
        dataframe: dataframe obtained from feature generation

    Returns: modified dataframe specific for this model
    '''

    dataframe['no_show_before_sq'] = dataframe['no_show_before'] ** (2)
    dataframe['sched_days_advanced_sq'] = dataframe['sched_days_advanced'] ** 2
    dataframe['age_sq'] = dataframe['age'] ** 2
    dataframe['distance_to_usz_sq'] = dataframe['distance_to_usz'] ** 2

    dataframe['sched_2_days'] = dataframe['sched_days_advanced'] <= 2
    dataframe['close_to_usz'] = dataframe['distance_to_usz'] < 16
    dataframe['age_20_60'] = (dataframe['age'] > 20) & (dataframe['age'] < 60)

    dummy = pd.get_dummies(dataframe['sex'])
    dataframe = pd.concat([dataframe, dummy], axis=1)

    return dataframe
