import altair as alt
import pandas as pd


def plot_hist_idle_gap_length(appts_and_gaps: pd.DataFrame) -> alt.Chart:
    """
    Plot a histogram of idle time gap lengths.

    Args:
        appts_and_gaps: result of `calc_appts_and_gaps`

    Returns: histogram of idle time gap lengths.

    """
    return alt.Chart(appts_and_gaps.loc[appts_and_gaps['status'] == 'idle', ['status_duration']]).mark_bar().encode(
        alt.X('status_duration', bin=alt.Bin(extent=[0, 1], step=0.05)),
        y='count()'
    )
