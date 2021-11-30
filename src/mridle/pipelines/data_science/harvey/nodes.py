import altair as alt
import pandas as pd
from typing import Dict


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
