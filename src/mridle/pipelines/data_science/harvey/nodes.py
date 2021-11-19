import altair as alt
import pandas as pd
from typing import Dict


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

    dataframe['hour_sched'].fillna(dataframe['hour_sched'].median(), inplace=True)

    dataframe = dataframe.dropna()

    return dataframe


def plot_harvey_metrics(model_results: Dict[str, pd.DataFrame]) -> alt.Chart:
    all_results = pd.DataFrame()
    for model_name, results_df in model_results.items():
        results_df = pd.melt(results_df, var_name='metric', id_vars=['partition'])
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
