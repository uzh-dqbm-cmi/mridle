import altair as alt


def plot_num_preds_vs_precision(results_df, n_preds_col='# No-show predictions per week',
                                precision_col='PPV / Precision', name_col='model_name'):
    """
    Taking in a dataframe containing evaluation results achieved by each model, produce a scatter plot with the
     number of positive predictions on the X-axis, and the precision on the Y-axis

    :param results_df: dataframe of model evaluations on a test set
    :param n_preds_col: column name containing the number of positive predictions a model gives
    :param precision_col: column name containing the precision a model achieves
    :param name_col: column name containing the name of the model to be used in the plot legend
    :return: altair scatter plot with number of positive predictions on the X-axis, and the precision on the Y-axis
    """
    return alt.Chart(results_df[[n_preds_col, precision_col, name_col]]).mark_circle(size=60, opacity=1).encode(
        x=n_preds_col,
        y=alt.Y(precision_col, scale=alt.Scale(domain=(0, 1))),
        color=name_col
    ).properties(width=500)
