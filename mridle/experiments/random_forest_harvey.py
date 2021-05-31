from mridle.experiment import ModelRun
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple

cols_for_modeling = ['no_show_before', 'no_show_before_sq', 'sched_days_advanced', 'hour_sched',
                     'distance_to_usz', 'age', 'close_to_usz', 'male', 'female', 'age_sq',
                     'sched_days_advanced_sq', 'distance_to_usz_sq', 'sched_2_days', 'age_20_60']

col_names_normalization = ['male']

# Number of treas in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider in splits
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Min num of samples needed to split a node
min_samples_split = [2, 4, 6, 8, 10]
# min num of samples needed at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# bootstrap
bootstrap = [True, False]

hyperparams = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


class HarveyModel_RandomForest(ModelRun):

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
        """
        Provides test data

        Args:
            model

        Returns:
            a dataframe with the appropriate columns to test the model
        """
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, len(cols_for_modeling))), columns=cols_for_modeling)
        df['noshow'] = np.where(df[cols_for_modeling[0]] > 50, 1, 0)
        return df


def process_features_for_model(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Changes variables for model optimization modifying feature_df

    Args:
        dataframe: dataframe obtained from feature generation

    Returns: modified dataframe specific for this model
    """
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
