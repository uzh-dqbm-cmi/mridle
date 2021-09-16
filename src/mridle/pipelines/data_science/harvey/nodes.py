import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mridle.pipelines.data_science import feature_engineering
from mridle.pipelines.data_engineering.ris.nodes import build_slot_df
from mridle.utilities.experiment import PartitionedExperiment, ModelRun
from typing import Dict, Tuple


def build_harvey_et_al_features_set(status_df: pd.DataFrame, include_id_cols=False) -> pd.DataFrame:
    """
    Builds a feature set that replicates the Harvey et al model as best we can.
    So far includes:
        - sched_days_advanced: Number of days the appt was scheduled in advance
        - day_of_week: The day of the week of the appt (1=Monday)
        - modality: The UniversalServiceName of the appt
        - marital: Zivilstand of the patient
        - distance_to_usz: distance from the patient's home address to the hospital, approximated from Post Codes
        - no_show_before: The number of no shows the patient has had up to the date of the appt
    Args:
        status_df:
        include_id_cols: Whether to remove the id columns

    Returns:

    """
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    status_df = feature_engineering.feature_hour_sched(status_df)
    status_df = feature_engineering.feature_days_scheduled_in_advance(status_df)
    status_df = feature_engineering.feature_day_of_week(status_df)
    status_df = feature_engineering.feature_modality(status_df)
    status_df = feature_engineering.feature_insurance_class(status_df)
    status_df = feature_engineering.feature_sex(status_df)
    status_df = feature_engineering.feature_age(status_df)
    status_df = feature_engineering.feature_marital(status_df)
    status_df = feature_engineering.feature_post_code(status_df)
    status_df = feature_engineering.feature_distance_to_usz(status_df)
    status_df = feature_engineering.feature_no_show_before(status_df)

    agg_dict = {
        'NoShow': 'min',
        'hour_sched': 'first',
        'sched_days_advanced': 'first',
        'modality': 'last',
        'insurance_class': 'last',
        'day_of_week': 'last',
        'sex': 'last',
        'age': 'last',
        'marital': 'last',
        'post_code': 'last',
        'distance_to_usz': 'last',
        'no_show_before': 'last',
    }

    slot_df = build_slot_df(status_df, agg_dict, include_id_cols=include_id_cols)

    return slot_df


def process_features_for_model(dataframe: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Changes variables for model optimization modifying feature_df

    Args:
        dataframe: dataframe obtained from feature generation
        parameters: parameters dictionary, namely which features to include

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

    dataframe['hour_sched'].fillna(dataframe['hour_sched'].median(), inplace=True)

    feature_list = parameters['models']['harvey']['features']
    feature_list.append(parameters['models']['harvey']['label_key'])
    features_df = dataframe[feature_list]

    return features_df


def train_harvey_model_random_forest(features_df: pd.DataFrame, params: Dict) -> Tuple[PartitionedExperiment, Dict]:
    model = RandomForestClassifier()

    hp_config = params['hyperparameters']
    # Number of treas in random forest
    n_estimators = [int(x) for x in np.linspace(**hp_config['n_estimators'])]
    # Number of features to consider in splits
    max_features = hp_config['max_features']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(**hp_config['max_depth'])]
    max_depth.append(None)
    # Min num of samples needed to split a node
    min_samples_split = hp_config['min_samples_split']
    # min num of samples needed at each leaf node
    min_samples_leaf = hp_config['min_samples_leaf']
    # bootstrap
    bootstrap = hp_config['bootstrap']

    hyperparams = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                   'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    exp = PartitionedExperiment(name=params['name'], data_set=features_df, model_run_class=ModelRun, model=model,
                                preprocessing_func=None, label_key=params['label_key'],
                                hyperparams=hyperparams, verbose=params['verbose'],
                                search_type=params['search_type'], scoring_fn=params['scoring_fn'])
    results = exp.run(run_hyperparam_search=params['run_hyperparam_search'])
    return exp, results
