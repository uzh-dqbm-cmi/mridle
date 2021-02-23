import altair as alt
import pandas as pd
import numpy as np


def calc_idle_time_gaps(dicom_times_df: pd.DataFrame) -> pd.DataFrame:
    idle_df = dicom_times_df.copy()
    idle_df['date'] = idle_df['image_start'].dt.date
    key_cols = ['date', 'image_device_id']
    idle_df = idle_df.sort_values(key_cols + ['image_start'])
    idle_df['previous_end_shift'] = idle_df.groupby(key_cols)['image_end'].shift(1)
    idle_df['previous_end'] = np.where(idle_df['previous_end_shift'] < idle_df['image_start'],
                                       idle_df['previous_end_shift'], pd.NaT)
    idle_df['previous_end'] = pd.to_datetime(idle_df['previous_end'])
    one_hour = pd.to_timedelta(1, unit='H')
    idle_df['idle_time'] = np.where(idle_df['previous_end'] < idle_df['image_start'],
                                    (idle_df['image_start'] - idle_df['previous_end']) / one_hour, 0)
    return idle_df


def calc_daily_idle_time_stats(idle_df: pd.DataFrame) -> pd.DataFrame:
    daily_idle_stats = idle_df.groupby(['date', 'image_device_id']).agg({
        'idle_time': 'sum',
        'image_start': 'min',
        'image_end': 'max'
    }).reset_index()
    one_hour = pd.to_timedelta(1, unit='H')
    daily_idle_stats['active_hours'] = (daily_idle_stats['image_end'] - daily_idle_stats['image_start']) / one_hour
    daily_idle_stats['idle_time_pct'] = daily_idle_stats['idle_time'] / daily_idle_stats['active_hours']
    return daily_idle_stats


def calc_appts_and_gaps(idle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a row-per-appoinmtent dataframe into a dataframe that has one row per appointment and one row per idle gap
     between appointments.

    Args:
        idle_df: DICOM data with one row per appointment. Dataframe must contain the columns ['date', 'image_device_id',
         'image_start', 'image_end', 'previous_end']

    Returns: A dataframe that has one row per appointment and one row per idle gap between appointments.

    """
    appts = idle_df[['date', 'image_device_id', 'image_start', 'image_end']].copy()
    appts.columns = ['date', 'image_device_id', 'start', 'end']
    appts['status'] = 'active'

    gaps = idle_df[['date', 'image_device_id', 'previous_end', 'image_start']].copy()
    gaps = gaps[~gaps['previous_end'].isna()].copy()
    gaps.columns = ['date', 'image_device_id', 'start', 'end']
    gaps = gaps[gaps['start'].dt.date == gaps['end'].dt.date].copy()
    gaps['status'] = 'idle'

    appts_and_gaps = pd.concat([appts, gaps])
    return appts_and_gaps


def plot_daily_appt_idle_segments(appts_and_gaps: pd.DataFrame) -> alt.Chart:
    """
    Plot a history of appointments, where each day is displayed as a row with colored segments indicating active and
     idle periods.

    Args:
        appts_and_gaps: result of `calc_appts_and_gaps`

    Returns: Figure where x-axis is time of day and y-axis is date. Each day-row is displayed as a row with colored
     segments indicating active and idle periods. Chart is faceted by image_device_id.

    """
    return alt.Chart(appts_and_gaps).mark_bar().encode(
        alt.X('hoursminutes(start)'),
        alt.X2('hoursminutes(end)'),
        y='date',
        color='status',
        tooltip=['date', 'hoursminutes(start)', 'hoursminutes(end)', 'status'],
    ).facet(
        column=alt.Row("image_device_id:N")
    )


def plot_hist_idle_gap_length(daily_idle_stats: pd.DataFrame) -> alt.Chart:
    """
    Plot a histogram of idle time gap lengths.

    Args:
        daily_idle_stats: result of `calc_daily_idle_time_stats`

    Returns: histogram of idle time gap lengths.

    """
    return alt.Chart(daily_idle_stats[['date', 'idle_time']]).mark_bar().encode(
        alt.X('idle_time', bin=alt.Bin(extent=[0, 1], step=0.05)),
        y='count()'
    )


def plot_total_active_idle_time_per_day(daily_idle_stats: pd.DataFrame) -> alt.Chart:
    """
    Plot the total hours spent active and idle for each day.

    Args:
        daily_idle_stats: result of `calc_daily_idle_time_stats`

    Returns: Figure where x-axis is date and y-axis is total hours. Each day-column is a stacked bar with total active
     and total idle hours for that day. The chart is faceted by image_device_id.

    """
    daily_between_times_melted = pd.melt(daily_idle_stats, id_vars=['date', 'image_device_id'],
                                         value_vars=['active_hours', 'idle_time'], var_name='time_type',
                                         value_name='hours')

    return alt.Chart(daily_between_times_melted).mark_bar().encode(
        alt.X("date"),
        y='hours',
        color='time_type:N',
    ).facet(
        column=alt.Row("image_device_id:N")
    )


def plot_pct_idle_per_day(daily_idle_stats: pd.DataFrame) -> alt.Chart:
    """
    Plot the percent of time the machine spends idle each day.

    Args:
        daily_idle_stats: result of `calc_daily_idle_time_stats`

    Returns: Figure where x-axis is date and y-axis is total hours. Each day-column displays the percent of time the
     machine was idle for that day. The chart is faceted by image_device_id.

    """
    return alt.Chart(daily_idle_stats).mark_bar().encode(
        alt.X("date"),
        y='idle_time_pct',
    ).facet(
        column=alt.Row("image_device_id:N")
    )
