import altair as alt
import pandas as pd
import numpy as np
import datetime


def calc_idle_time_gaps(dicom_times_df: pd.DataFrame, tp_agg_df: pd.DataFrame, time_buffer_mins=0) -> pd.DataFrame:
    """
    Calculate the length of idle time gaps in between appointments.

    Args:
        dicom_times_df: result of `mridle.data_management.format_dicom_times_df`
        tp_agg_df: result of `mridle.data_management.aggregate_terminplanner`
        time_buffer_mins: buffer time in minutes which is taken from the start and added to end of each appointment

    Returns: `dicom_times_df` dataframe with added columns:
     - `previous_end`: the end time of the preceding appointment (if the first appointment of the day, then pd.NaT)
     - `day_start_tp`: 'official' start of day as in terminplanner data
     - `day_end_tp`: 'official' end of day as in terminplanner data
     - `first_appt`: flag indicating if the appointment is the first one for the given weekday and machine combination
     - `last_appt`: flag indicating if the appointment is the last one for the given weekday and machine combination
     - ... and others used to aid buffer & idle time calculation later
    """
    idle_df = dicom_times_df.copy()

    idle_df['date'] = pd.to_datetime(idle_df['image_start'].dt.date)
    idle_df['day_of_week'] = idle_df['image_start'].dt.day_name()

    # Join on terminplanner data
    idle_df = idle_df.merge(tp_agg_df, how='left', on=['day_of_week', 'image_device_id'])
    idle_df = idle_df[(idle_df['image_start'] >= idle_df["applicable_from"]) &
                      (idle_df['image_start'] <= idle_df["applicable_to"])]
    idle_df = idle_df.drop(['applicable_from', 'applicable_to'], axis=1)

    # Using terminplanner df, add flag for each appointment indicating whether it falls within the times outlined by the
    # terminplanner, and then limit our data to only those appts
    idle_df['within_day'] = np.where(
        (idle_df['image_end'].dt.time > idle_df['day_start_tp']) &
        (idle_df['image_start'].dt.time < idle_df['day_end_tp']),
        1, 0)

    idle_df = idle_df[idle_df['within_day'] == 1]

    idle_df['day_start_tp'] = idle_df.apply(lambda x: datetime.datetime.combine(x['date'], x['day_start_tp']), axis=1)
    idle_df['day_end_tp'] = idle_df.apply(lambda x: datetime.datetime.combine(x['date'], x['day_end_tp']), axis=1)

    # Add columns indicating if the appointment was the first / last appointment for that day for that MR machine
    idle_df['first_appt'] = idle_df.groupby(['date', 'image_device_id'])['image_start'].transform('rank',
                                                                                                  ascending=True)
    idle_df['first_appt'] = np.where(idle_df['first_appt'] == 1, 1, 0)
    idle_df['last_appt'] = idle_df.groupby(['date', 'image_device_id'])['image_start'].transform('rank',
                                                                                                 ascending=False)
    idle_df['last_appt'] = np.where(idle_df['last_appt'] == 1, 1, 0)

    idle_df['one_side_buffer_flag'] = 0
    # For the 'last appts' in the day which finish within the day (by terminplanner), we need to calculate idle time
    # until the end of the day as given by the terminplanner, so we add a dummy appointment row for the end of the day,
    # which will then be used automatically later on for calculating idle/buffer
    #
    # If appt end is later than the end of day by the terminplanner, then we need to note this for later (for
    # buffer calc);
    # we also don't need to add a dummy appointment row
    last_appts = idle_df[idle_df['last_appt'] == 1]
    new_rows_df = pd.DataFrame(columns=['image_start', 'image_end', 'date', 'image_device_id', 'within_day',
                                     'day_length_tp', 'day_start_tp', 'day_end_tp', 'one_side_buffer_flag'])

    for idx, row in last_appts.iterrows():
        if row['image_end'] < row['day_end_tp']:
            new_rows_df = new_rows_df.append({
                'image_start': row['day_end_tp'],
                'image_end': row['day_end_tp'],
                'date': row['date'],
                'image_device_id': row['image_device_id'],
                'within_day': 1,
                'day_length_tp': row['day_length_tp'],
                'day_start_tp': row['day_start_tp'],
                'day_end_tp': row['day_end_tp'],
                'one_side_buffer_flag': 1
            }, ignore_index=True)
        else:
            idle_df.loc[idle_df['AccessionNumber'] == row['AccessionNumber'], 'one_side_buffer_flag'] = 1

    idle_df = pd.concat([idle_df, new_rows_df])

    key_cols = ['date', 'image_device_id']
    idle_df = idle_df.sort_values(key_cols + ['image_start'])
    idle_df['previous_end_shift'] = idle_df.groupby(key_cols)['image_end'].shift(1)

    # if there is overlap between the appointments (previous end time is after current start time), then ignore this
    # 'between' segment
    idle_df['previous_end'] = np.where(idle_df['previous_end_shift'] < idle_df['image_start'],
                                       idle_df['previous_end_shift'], pd.NaT)
    idle_df['previous_end'] = pd.to_datetime(idle_df['previous_end'])
    one_hour = pd.to_timedelta(1, unit='H')

    # Similar to getting the last appts for the day above, here we get the first appts in the day so we can calculate
    # idle time before this appt (i.e. from start of day as given by terminplanner to start of first appt,
    # is counted as idle)
    # Here we just set the 'end' of the 'previous_appt' to be the start of the day by TP (if first appt started after
    # the start of the day), and idle/buffer time will be calculated as normal later
    first_appts = idle_df[idle_df['first_appt'] == 1]
    for idx, row in first_appts.iterrows():
        if row['image_start'] > row['day_start_tp']:
            idle_df.loc[idle_df['AccessionNumber'] == row['AccessionNumber'], 'previous_end'] = row['day_start_tp']
        else:
            idle_df.loc[idle_df['AccessionNumber'] == row['AccessionNumber'], 'one_side_buffer_flag'] = 1

    idle_df['time_between_appt'] = (idle_df['image_start'] - idle_df['previous_end']) / one_hour

    time_buffer_dt = pd.to_timedelta(time_buffer_mins, unit='minute')

    # If time between appointments is larger than 2 'buffer times' (one before and one after each appointment), then
    # set buffer_time to be 2 * user-specified buffer time. If less, then it means there's overlapping appts with buffer
    # time included, so we set all the time_between_appt to be buffer time (zero idle time is dealt with in line above)
    idle_df['buffer_time'] = np.minimum(idle_df['time_between_appt'], time_buffer_dt * 2 / one_hour)

    # Add buffer cols to idle_df
    idle_df = idle_df.apply(add_buffer_cols, axis=1)
    idle_df.drop('buffer_time', axis=1, inplace=True)
    # If appt row is one of the dummy 'end of day' appointments created above, set the buffer (which is calculated
    # later) to be 0 by setting the image_start_buffer time back to the original end of day time
    idle_df.loc[(idle_df['one_side_buffer_flag'] == 1) & (idle_df['AccessionNumber'].isna()),
                'image_start_buffer'] = idle_df.loc[(idle_df['one_side_buffer_flag'] == 1) &
                                                    (idle_df['AccessionNumber'].isna()), 'image_start']
    # Similarly, if appt row is the first appointments in the day (and it also start after the TP start of day), then
    # set the buffer (which is calculated later) to be 0 by setting the previous_end_buffer
    # time back to the original start of day time

    idle_df.loc[(idle_df['one_side_buffer_flag'] == 0) & (idle_df['first_appt'] == 1),
                'previous_end_buffer'] = idle_df.loc[(idle_df['one_side_buffer_flag'] == 0) &
                                                     (idle_df['first_appt'] == 1), 'previous_end']

    return idle_df


def add_buffer_cols(appt_row: pd.Series) -> pd.Series:
    """
    Designed to be used row-wise (e.g. in a pd.apply() function, using the idle_df dataframe.

    Take in row from df with columns 'buffer_time', 'previous_end', 'image_start', and return a row with two columns
    added, namely: previous_end_buffer and image_start_buffer

    Args:
        appt_row: one row from df with columns 'buffer_time', 'previous_end'

    Returns: row with two columns added, namely: previous_end_buffer and image_start_buffer

    """

    buffer_per_appt = pd.to_timedelta(appt_row['buffer_time'] / 2, unit='H')
    appt_row['previous_end_buffer'] = appt_row['previous_end'] + buffer_per_appt \
        if not pd.isnull(appt_row['previous_end']) else appt_row['previous_end']
    appt_row['image_start_buffer'] = appt_row['image_start'] - buffer_per_appt \
        if not pd.isnull(appt_row['image_start']) else appt_row['image_start']

    return appt_row


def calc_daily_idle_time_stats(appts_and_gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a row-per-appointment dataframe into a row-per-day dataframe showing active and idle time per day.
    Args:
        appts_and_gaps: resulting df from calc_appts_and_gaps()
    Returns: Dataframe with columns ['date', 'image_device_id', 'idle' (float hours), 'buffer' (float hours),
     'start' (time of start of the day), 'end' (time of end of the day), 'total_day_time' (float hours),
     active' (float hours), 'active_pct', 'idle_pct', 'buffer_pct']

    """
    appts_and_gaps_copy = appts_and_gaps.copy()
    daily_stats = appts_and_gaps_copy.groupby(['date', 'image_device_id', 'status']).agg({
        'status_duration': 'sum'
    }).reset_index()

    one_hour = pd.to_timedelta(1, unit='H')

    total_day_times = appts_and_gaps_copy.groupby(['date', 'image_device_id']).agg({
        'start': 'min',
        'end': 'max'
    }).reset_index()

    total_day_times['total_day_time'] = (total_day_times['end'] - total_day_times['start']) / one_hour

    stats = daily_stats.pivot_table(columns='status', values='status_duration',
                                    index=['date', 'image_device_id']).reset_index()
    stats = stats.merge(total_day_times, on=['date', 'image_device_id'])

    # We calculate active time this way as there might be overlaps in appts, so we don't want to doublecount
    stats['active'] = stats['total_day_time'] - stats['buffer'] - stats['idle']
    stats['active_pct'] = stats['active'] / stats['total_day_time']
    stats['buffer_pct'] = stats['buffer'] / stats['total_day_time']
    stats['idle_pct'] = stats['idle'] / stats['total_day_time']

    return stats


def calc_appts_and_gaps(idle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a row-per-appointment dataframe into a dataframe that has one row per appointment and one row per idle gap
     between appointments.

    Args:
        idle_df: DICOM data with one row per appointment. Dataframe must contain the columns ['date', 'image_device_id',
         'image_start', 'image_end', 'previous_end', 'image_start_buffer', 'previous_end_buffer']

    Returns: A dataframe that has one row per appointment, one row per idle gap between appointments, and a row each
    for the buffer time before and after an appointment.

    """
    idle_df_copy = idle_df.copy()
    idle_df_copy = idle_df_copy[idle_df_copy['within_day'] == 1]

    appts = idle_df_copy[['date', 'image_device_id', 'image_start', 'image_end']].copy()
    appts.columns = ['date', 'image_device_id', 'start', 'end']
    appts['status'] = 'active'

    gaps = idle_df_copy[['date', 'image_device_id', 'previous_end_buffer', 'image_start_buffer']].copy()
    gaps = gaps[~gaps['previous_end_buffer'].isna()].copy()
    gaps.columns = ['date', 'image_device_id', 'start', 'end']
    gaps = gaps[gaps['start'].dt.date == gaps['end'].dt.date].copy()
    gaps['status'] = 'idle'

    post_buffers = idle_df_copy[['date', 'image_device_id', 'previous_end', 'previous_end_buffer']].copy()
    post_buffers = post_buffers[~post_buffers['previous_end_buffer'].isna()].copy()
    post_buffers.columns = ['date', 'image_device_id', 'start', 'end']
    post_buffers = post_buffers[post_buffers['start'].dt.date == post_buffers['end'].dt.date].copy()
    post_buffers['status'] = 'buffer'

    pre_buffers = idle_df_copy[['date', 'image_device_id', 'image_start_buffer', 'image_start']].copy()
    pre_buffers = pre_buffers[~pre_buffers['image_start_buffer'].isna()].copy()
    pre_buffers.columns = ['date', 'image_device_id', 'start', 'end']
    pre_buffers = pre_buffers[pre_buffers['start'].dt.date == pre_buffers['end'].dt.date].copy()
    pre_buffers['status'] = 'buffer'

    appts_and_gaps = pd.concat([appts, gaps, post_buffers, pre_buffers])
    appts_and_gaps['status_duration'] = (appts_and_gaps['end'] - appts_and_gaps['start']) / pd.to_timedelta(1, unit='H')
    appts_and_gaps = appts_and_gaps.sort_values(['start', 'end'])

    return appts_and_gaps


def plot_daily_appt_idle_segments(appts_and_gaps: pd.DataFrame, height: int = 300, bar_size: int = 5,
                                  width: int = 300) -> alt.Chart:
    """
    Plot a history of appointments, where each day is displayed as a row with colored segments indicating active and
     idle periods.

    Args:
        appts_and_gaps: result of `calc_appts_and_gaps`
        height: Height of plot window. Default value 300
        width: Width of plot window. Default value 300
        bar_size: size of bars in plot. Default value 5
    Returns: Figure where x-axis is time of day and y-axis is date. Each day-row is displayed as a row with colored
     segments indicating active and idle periods. Chart is faceted by image_device_id.

    """
    domain = ['Active', 'Idle', 'Buffer']
    range_ = ['#0065af', '  #fe8126  ', ' #fda96b ']

    plot_data = appts_and_gaps.copy()
    plot_data.rename(columns={'status': 'Machine Status'}, inplace=True)
    plot_data["Machine Status"].replace({'active': 'Active', 'idle': 'Idle', 'buffer': 'Buffer'}, inplace=True)

    return alt.Chart(plot_data).mark_bar(size=bar_size).encode(
        alt.X('hoursminutes(start)', title="Time of day"),
        alt.X2('hoursminutes(end)', title=""),
        alt.Y('yearmonthdate(date)', axis=alt.Axis(grid=False), title="Date"),
        color=alt.Color('Machine Status', scale=alt.Scale(domain=domain, range=range_)),
        tooltip=['date', 'hoursminutes(start)', 'hoursminutes(end)', 'Machine Status'],
    ).properties(height=height, width=width
                 ).facet(
        column=alt.Row("image_device_id:N", title="MR Machine #")
    )


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


def plot_total_active_idle_buffer_time_per_day(daily_idle_stats: pd.DataFrame,
                                               use_percentage: bool = False) -> alt.Chart:
    """
    Plot the total hours spent active and idle for each day.

    Args:
        daily_idle_stats: result of `calc_daily_idle_time_stats`
        use_percentage: boolean indicating whether to plot the y-axis as a percentage of the total day, or using
        absolute time (hours)

    Returns: Figure where x-axis is date and y-axis is total hours. Each day-column is a stacked bar with total active,
     total idle, and total buffer hours for that day. The chart is faceted by image_device_id.

    """
    if use_percentage:
        val_vars = ['active_pct', 'idle_pct', 'buffer_pct']
        y_label = "Percentage of day"
    else:
        val_vars = ['active', 'idle', 'buffer']
        y_label = "Hours"

    daily_between_times_melted = pd.melt(daily_idle_stats, id_vars=['date', 'image_device_id'],
                                         value_vars=val_vars, var_name='Machine Status',
                                         value_name='hours')

    daily_between_times_melted["Machine Status"].replace(
        {val_vars[0]: 'Active', val_vars[1]: 'Idle', val_vars[2]: 'Buffer'}, inplace=True)

    domain = ['Active', 'Idle', 'Buffer']
    range_ = ['#0065af', '#fe8126', '#fda96b']

    return alt.Chart(daily_between_times_melted).mark_bar().encode(
        alt.X("date", axis=alt.Axis(title="Date")),
        y=alt.Y('hours', axis=alt.Axis(title=y_label)),
        color=alt.Color('Machine Status:N', scale=alt.Scale(domain=domain, range=range_)),
        tooltip=['date', 'hours'],
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
        y='idle_pct',
    ).facet(
        column=alt.Row("image_device_id:N")
    )
