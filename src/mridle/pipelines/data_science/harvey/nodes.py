import altair as alt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mridle.utilities.experiment import PartitionedExperiment, ModelRun
from typing import Dict, Tuple


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

    return dataframe


def train_harvey_model_logistic_reg(features_df: pd.DataFrame, params: Dict) -> Tuple[PartitionedExperiment, Dict]:
    model = LogisticRegression()
    exp = PartitionedExperiment(name=params['name'], data_set=features_df, feature_subset=params['features'],
                                model_run_class=ModelRun, model=model, preprocessing_func=None,
                                label_key=params['label_key'], hyperparams=params['hyperparameters'],
                                verbose=params['verbose'], search_type=params['search_type'],
                                scoring_fn=params['scoring_fn'])
    results = exp.run(run_hyperparam_search=params['run_hyperparam_search'])
    return exp, results


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

    exp = PartitionedExperiment(name=params['name'], data_set=features_df, feature_subset=params['features'],
                                model_run_class=ModelRun, model=model, preprocessing_func=None,
                                label_key=params['label_key'], hyperparams=hyperparams, verbose=params['verbose'],
                                search_type=params['search_type'], scoring_fn=params['scoring_fn'])
    results = exp.run(run_hyperparam_search=params['run_hyperparam_search'])
    return exp, results


def plot_harvey_metrics(model_results: Dict[str, Dict]) -> alt.Chart:
    all_results = pd.DataFrame()
    for model_name, results in model_results.items():
        results_df = pd.DataFrame(results)
        results_df = pd.melt(results_df, var_name='metric')
        results_df['model'] = model_name
        all_results = pd.concat([all_results, results_df])

    chart = alt.Chart(all_results).mark_circle(size=60, opacity=0.7).encode(
        x='model',
        y='value',
        color='model',
    ).facet(
        column='metric'
    )
    return chart
