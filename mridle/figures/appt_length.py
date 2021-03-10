import altair as alt
import pandas as pd


def plot_appt_len_vs_var(dicom_df: pd.DataFrame, variable: str, plot_type: str, group_col: str = None,
                         properties: dict = {'width': 200, 'height': 200}, sort_order: list = None) -> alt.Chart:
    """
    Generate a plot for a given variable against appt_len_float, with these on the x- and y-axis respectively.
    Can supply a column to group the x-axis data by, with a new plot being created for each group in this column
    Plot can be scatterplot or a boxplot
    Sort order of x-axis categories and properties for plot window can be specified

    Args:
        dicom_df: dataframe with dicom data, requires appt_len_float column to be present, as well as those specified
        in the 'variable' and the 'group_col' parameters
        variable: column/variable to plot against appt_len_float; will be on x-axis
        plot_type: scatter or boxplot
        group_col: optional column to group data on x-axis by. Will create new plot for each implied group present
        properties: height and width for the plot
        sort_order: sort order of 'variable' levels - esp. used for day_of_week (sort_order=['Monday', 'Tuesday', ...]

    Returns: Plots requested chart in window, using altair

    """
    if sort_order is None:
        sort_order = []

    data_cols = [variable, 'appt_len_float']
    if group_col:
        data_cols.append(group_col)

    if plot_type == "scatter":
        fig = alt.Chart(dicom_df[data_cols]).mark_point().encode(
            alt.X(variable, sort=sort_order),
            y='appt_len_float'
        )

    elif plot_type == "boxplot":
        fig = alt.Chart(dicom_df[data_cols]).mark_boxplot().encode(
            alt.X(variable, sort=sort_order),
            y='appt_len_float'
        )
    else:
        raise ValueError("No plot generated - make sure 'plot_type' argument is either 'scatter' or 'boxplot")

    fig = fig.properties(width=properties['width'], height=properties['height'])

    if group_col:
        fig = fig.facet(column=group_col)

    return fig
