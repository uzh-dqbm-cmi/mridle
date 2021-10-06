"""
Code to process the DICOM metadata extracts from the DFL-IT. This data extract comes with each appointment having
multiple rows associated with it, each row representing an image scanned by the MRI machine. We receive just a subset
of the metadata related to this image, and no actual image file.

The below functions are used in the data preprocessing step(s), and individual descriptions are given at the start
of each function.
"""
import pandas as pd
import datetime as dt
import numpy as np
import datetime
import altair as alt


def preprocess_dicom_data(df, id_list_df):
    dicom_5_years = df.copy()
    dicom_5_years = subset_valid_appts(dicom_5_years, id_list_df)
    dicom_5_years = subset_machines(dicom_5_years)
    dicom_5_years = remove_na_and_duplicates(dicom_5_years)
    dicom_5_years = process_date_cols(dicom_5_years)
    dicom_5_years = add_image_time_cols(dicom_5_years)
    dicom_5_years = remove_gaps_at_start_end(dicom_5_years)

    return dicom_5_years


def aggregate_dicom_images(df):
    df_copy = df.copy()
    df_copy_agg = df_copy.groupby(['AccessionNumber', 'big_image_gap', 'StationName']).agg(
        {'acq_datetime': [min, max]}).reset_index()
    df_copy_agg.columns = ['AccessionNumber', 'big_image_gap', 'image_device_id', 'image_start', 'image_end']
    dicom_data = df_copy_agg[['AccessionNumber', 'image_device_id', 'image_start', 'image_end']]
    return dicom_data


def integrate_dicom_data(slot_df: pd.DataFrame, dicom_times_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate DICOM data into slot_df to update the appointment information with more reliable information.
    - For all appointments, the device_id is updated with the one listed in dicom_times_df (the device actually used, as
        opposed to the device that was planned to be used).
    - For show and inpatient appointments, the appointment start and end times are updated to the accurate DICOM values
        (the actual imaging start and end times, as opposed to the planned appt start time).
    Original values from the status_df are maintained in separate columns: status_start, status_end, device_from_status.
    Args:
        slot_df: row-per-status-change dataframe.
        dicom_times_df: A dataframe of DICOM metadata information, with the columns:
            - AccessionNumber
            - image_device_id
            - image_start
            - image_end
    Returns: slot_df with new and modified time and device columns. Columns include:
        - start_time (as determined by DICOM data)
        - status_start (start time as originally stated by status data)
        - end_time (as determined by DICOM data)
        - status_end (end time as originally stated by status data)
        - EnteringOrganisationDeviceID (device as determined by DICOM data)
        - device_from_status (device as originally stated by status data)
    Raises:
        ValueError if number of rows in status_df changes during this transformation
    """
    slot_w_dicom_df = pd.merge(slot_df, dicom_times_df, how='left',
                               left_on=['FillerOrderNo', 'EnteringOrganisationDeviceID'],
                               right_on=['AccessionNumber', 'image_device_id'])

    # move times defined by status changes to separate columns to allow overwriting the original columns with dicom data
    slot_w_dicom_df['status_start'] = slot_w_dicom_df['start_time']
    slot_w_dicom_df['status_end'] = slot_w_dicom_df['end_time']

    # for show and in-patient appointments, use dicom data for start and end times
    slot_w_dicom_df['start_time'] = slot_w_dicom_df.apply(update_start_time_col_from_dicom, axis=1)
    slot_w_dicom_df['end_time'] = slot_w_dicom_df.apply(update_end_time_col_from_dicom, axis=1)

    # update device used
    slot_w_dicom_df['device_from_status'] = slot_w_dicom_df['EnteringOrganisationDeviceID']
    slot_w_dicom_df['EnteringOrganisationDeviceID'] = slot_w_dicom_df.apply(update_device_id_from_dicom, axis=1)

    if slot_df.shape[0] != slot_w_dicom_df.shape[0]:
        raise ValueError('Number of rows in slot_w_dicom_df ({:,.0f}) does not match original slot_df ({:,.0f})'.format(
            slot_w_dicom_df.shape[0], slot_df.shape[0]))

    return slot_w_dicom_df


def prep_terminplanner(terminplanner_df):
    """

    Args:
        terminplanner_df:

    Returns:

    """
    tp_agg_df = aggregate_terminplanner(terminplanner_df)
    import datetime
    new_rows = pd.DataFrame(
        [[1, 'Monday', 'MR1 IDR (Montag)', '17.05.2020', '01.01.2055', datetime.time(7, 0), datetime.time(18, 0), 660],
         [2, 'Monday', 'MR2 IDR (Montag)', '17.05.2020', '01.01.2055', datetime.time(7, 0), datetime.time(18, 0), 660],
         [1, 'Tuesday', 'MR1 IDR (Dienstag)', '19.10.2020', '01.01.2055', datetime.time(7, 0), datetime.time(18, 0),
          660],
         [2, 'Tuesday', 'MR2 IDR (Dienstag)', '30.06.2020', '01.01.2055', datetime.time(7, 0), datetime.time(18, 0),
          660],
         [1, 'Wednesday', 'MR1 IDR (Mittwoch)', '19.10.2020', '01.01.2055', datetime.time(7, 0), datetime.time(18, 0),
          660],
         [1, 'Thursday', 'MR1 IDR (Donnerstag)', '30.06.2020', '01.01.2055', datetime.time(7, 0), datetime.time(20, 30),
          805],
         [2, 'Thursday', 'MR2 IDR (Donnerstag)', '31.10.2019', '01.01.2055', datetime.time(7, 0), datetime.time(20, 30),
          805],
         ], columns=tp_agg_df.columns)
    tp_agg_df = tp_agg_df.append(new_rows)
    tp_agg_df = tp_agg_df.reset_index(drop=True)
    return tp_agg_df


def generate_idle_time_stats(dicom_times_df: pd.DataFrame, terminplanner_aggregated_df: pd.DataFrame):
    """

    Args:
        dicom_times_df:
        terminplanner_aggregated_df:

    Returns:

    """
    idle_df = calc_idle_time_gaps(dicom_times_df, terminplanner_aggregated_df, time_buffer_mins=5)
    appts_and_gaps = calc_appts_and_gaps(idle_df)
    daily_idle_stats = calc_daily_idle_time_stats(appts_and_gaps)

    return appts_and_gaps, daily_idle_stats


def generate_plots(appts_and_gaps, daily_idle_stats):
    """

    Args:
        appts_and_gaps:
        daily_idle_stats:

    Returns:

    """

    alt.data_transformers.disable_max_rows()
    appts_and_gaps['date'] = pd.to_datetime(appts_and_gaps['date'])
    appts_and_gaps['start'] = pd.to_datetime(appts_and_gaps['start'])
    appts_and_gaps['end'] = pd.to_datetime(appts_and_gaps['end'])
    daily_idle_stats['start'] = pd.to_datetime(daily_idle_stats['start'])
    daily_idle_stats['end'] = pd.to_datetime(daily_idle_stats['end'])

    daily_idle_stats_mr1 = daily_idle_stats[daily_idle_stats['image_device_id'] == 1]
    day_summary_plot = plot_total_active_idle_buffer_time_per_day(daily_idle_stats_mr1, use_percentage=True)

    appts_and_gaps_mr1 = appts_and_gaps[appts_and_gaps['image_device_id'] == 1]
    full_zebra = plot_daily_appt_idle_segments(appts_and_gaps_mr1, width=500, height=5000)

    one_week = appts_and_gaps_mr1[(appts_and_gaps_mr1['start'].dt.date > pd.to_datetime('2019-03-30'))
                                  & (appts_and_gaps_mr1['end'].dt.date < pd.to_datetime('2019-04-06'))].copy()

    one_week_zebra = plot_daily_appt_idle_segments(one_week, bar_size=25, width=500, height=200)

    return day_summary_plot, full_zebra, one_week_zebra


# Helper functions
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

    return_chart = alt.Chart(daily_between_times_melted).mark_bar().encode(
        alt.X("date", axis=alt.Axis(title="Date")),
        y=alt.Y('hours', axis=alt.Axis(title=y_label), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Machine Status:N', scale=alt.Scale(domain=domain, range=range_)),
        tooltip=['date', 'hours'],
    ).properties(
        width=500
    ).facet(
        column=alt.Row("image_device_id:N")
    )

    return_chart = return_chart.configure_legend(
        labelFontSize=15,
        titleFontSize=20
    )

    return_chart = return_chart.configure_axis(
        titleFontSize=16,
        labelFontSize=14
    )

    return return_chart


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
    ).properties(
        height=height, width=width
    ).facet(
        column=alt.Row("image_device_id:N", title="MR Machine #")
    ).configure_axis(
        titleFontSize=19,
        labelFontSize=16,
        titleFontWeight="normal"
    ).configure_legend(
        labelFontSize=15,
        titleFontSize=20
    )


def subset_valid_appts(df, id_list_df):
    df_copy = df.copy()
    df_copy['AccessionNumber'] = pd.to_numeric(df_copy['AccessionNumber'], errors='coerce')
    df_copy = df_copy[~df_copy['AccessionNumber'].isna()]

    id_list_df = id_list_df.drop_duplicates()
    id_list_df_pivot = id_list_df.pivot(['SeriesUID', 'SopImageUid', 'StudyUID'], columns='tag_name',
                                        values='Value').reset_index()
    id_list_df_pivot['AcquisitionDate'] = id_list_df_pivot['AcquisitionDate'].astype(float)
    id_list_df_pivot['AccessionNumber'] = pd.to_numeric(id_list_df_pivot['AccessionNumber'], errors='coerce')
    id_list_df_pivot = id_list_df_pivot[~id_list_df_pivot['AccessionNumber'].isna()]

    id_list = id_list_df_pivot['AccessionNumber'].unique()
    df_copy = df_copy[df_copy['AccessionNumber'].isin(id_list)]
    return df_copy


def subset_machines(df):
    df_copy = df.copy()
    df_copy = df_copy[df_copy['StationName'].isin(['MT00000173', 'MT00000213'])]
    df_copy['StationName'] = df_copy['StationName'].map({'MT00000173': '1', 'MT00000213': '2'})

    return df_copy


def process_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take in the DICOM dataframe and add some date columns for easier calculations later

    Args:
        df: Dataframe where each row represents one image for an appointment,
         containing columns: ['AcquisitionDate', 'AcquisitionTime']

    Returns:
        dataframe with some reformatted date columns, and some new columns added
    """
    df_copy = df.copy()

    df_copy['AcquisitionTime'] = df_copy['AcquisitionTime'].apply(lambda a: "{:013.6F}".format(float(a)))
    df_copy['AcquisitionDate'] = pd.to_datetime(df_copy['AcquisitionDate'], format='%Y%m%d').dt.date
    df_copy['AcquisitionTime'] = pd.to_datetime(df_copy['AcquisitionTime'], format='%H%M%S.%f').dt.time
    df_copy.loc[~df_copy['AcquisitionTime'].isnull(),
                'acq_datetime'] = df_copy[~df_copy['AcquisitionTime'].isnull()].apply(
        lambda x: dt.datetime.combine(x['AcquisitionDate'], x['AcquisitionTime']), axis=1)

    return df_copy


def remove_na_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows and removes appointments with no AccessionNumber (appointment ID) or no data given

    TODO: Determine the cause of Null AccessionNumbers

    Args:
        df: Dataframe containing columns ['AccessionNumber', 'AcquisitionDate']
    Returns:
        Dataframe with duplicates removed and rows with null AccessionNumber or AcquisitionData columns

    """
    df_copy = df.copy()
    df_copy = df_copy.drop_duplicates()
    df_copy = df_copy[~df_copy['AccessionNumber'].isna()]
    df_copy = df_copy[~df_copy['AcquisitionDate'].isnull()]

    return df_copy


def add_image_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns describing the order (rank) and time differences between images within an appointment to the dataframe.

    The first group of these columns is time information for the previous and the next image in the sequence of images
    per appointment. Also added are columns calculating the time between the previous image in the appointment and the
    row image, as well as the time between the row image and the next image. (row image meaning the image that the
    given row is related to). Finally for this group, a flag is added for each image in the appointment if two images
    in the sequence are more than 1800 seconds / 30 minutes apart.

    The second group of columns is the 'rank' of the image, or what position it is in the sequence of images for the
    appointment (i.e. is it the first image, second, ..., 20th, ..., second last, last). And the same column idea
    is provided in a separate column, but with the rank in reverse order - this is mostly for easier calculations later
    and to make it clear which images are really the last (rather than just showing that they are the 254th image in a
    sequence of 254 images).

    Args:
        df: Dataframe where each row represents one image for an appointment,
         containing columns: ['AccessionNumber', 'AcquisitionTime', 'acq_datetime']

    Returns:
        Dataframe with the columns as described above added.

    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(['AccessionNumber', 'acq_datetime'])

    df_copy['AcquisitionTime_prev'] = df_copy.groupby('AccessionNumber')['AcquisitionTime'].shift(1)
    df_copy['acq_prev_datetime'] = df_copy.groupby('AccessionNumber')['acq_datetime'].shift(1)
    df_copy['acq_next_datetime'] = df_copy.groupby('AccessionNumber')['acq_datetime'].shift(-1)

    one_second = pd.to_timedelta(1, unit='S')
    df_copy['time_between_next_image'] = (df_copy['acq_next_datetime'] - df_copy['acq_datetime']) / one_second
    df_copy['time_between_prev_image'] = (df_copy['acq_datetime'] - df_copy['acq_prev_datetime']) / one_second

    df_copy['big_image_gap'] = np.max(df_copy[['time_between_next_image', 'time_between_prev_image']], axis=1)
    df_copy['big_image_gap'] = df_copy[['AccessionNumber', 'big_image_gap']].groupby('AccessionNumber').transform(
        lambda x: 1 if np.max(x) > 1800 else 0)

    df_copy['img_rank'] = df_copy.groupby('AccessionNumber')['acq_datetime'].transform('rank', ascending=True)
    df_copy['img_rank_rev'] = df_copy.groupby('AccessionNumber')['acq_datetime'].transform('rank', ascending=False)

    return df_copy


def remove_gaps_at_start_end(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there is a large gap (default of 30 minutes, 1800 seconds) contained within the first or last 5 images of the
    sequence, remove the images before/after this gap (before if it's at the start of an sequence, after if it's at
    the end).

    e.g. if there are 3 images taken at ~9am, and then a gap of 90 minutes until the remaining images are taken as part
    of this AccessionNumber (could be 100+ images), we remove these 3 images at the start, leaving behind only the
    'true' section of the appointment. It is assumed that these original 3 images were created as a computer error,
    and are not genuinely relating to the appointment.

    - Requires columns that are added by the get_image_time_cols() function

    Args:
        df: Dataframe with multiple rows per AccessionNumber (each row is the metadata for an individual image)

    Returns:
        Dataframe with rows removed as necessary.

    """
    df_copy = df.copy()
    remove_before = df_copy.loc[(df_copy['img_rank'] <= 5) & (df_copy['time_between_next_image'] > 1800),
                                ["AccessionNumber", "img_rank"]]
    remove_after = df_copy.loc[(df_copy['img_rank_rev'] <= 5) & (df_copy['time_between_prev_image'] > 1800),
                               ["AccessionNumber", "img_rank_rev"]]

    for idx, row in remove_before.iterrows():
        df_copy = df_copy[~((df_copy['img_rank'] <= row['img_rank']) &
                            (df_copy['AccessionNumber'] == row['AccessionNumber']))]

    for idx, row in remove_after.iterrows():
        df_copy = df_copy[~((df_copy['img_rank_rev'] <= row['img_rank_rev']) &
                            (df_copy['AccessionNumber'] == row['AccessionNumber']))]

    return df_copy


def update_start_time_col_from_dicom(row):
    if row['slot_type'] in ['show', 'inpatient'] and row['image_start'] is not None:
        return row['image_start']
    return row['status_start']


def update_end_time_col_from_dicom(row):
    if row['slot_type'] in ['show', 'inpatient'] and row['image_end'] is not None:
        return row['image_end']
    return row['status_end']


def update_device_id_from_dicom(row):
    if pd.isna(row['image_device_id']):
        return row['EnteringOrganisationDeviceID']
    else:
        return 'MR{}'.format(int(row['image_device_id']))


def aggregate_terminplanner(terminplanner_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in raw terminplanner data, where each row represents one possible appointment slot
    (e.g.   Terminbuch  Wochentag   TERMINRASTER_NAME      gültig von  gültig bis      Termin      Dauer in Min.
            MR1         DO          MR1 IDR (Donnerstag)   05.12.2018  20.02.2019      07:00       35
            MR1         DO          MR1 IDR (Donnerstag)   05.12.2018  20.02.2019      07:35       35

    )
    and returns an aggregated representation of this data, with each row representing one day of the week, and the
    start and end time of the window for appointments.

    Args:
        terminplanner_df: Raw terminplanner data, provided by Beat Hümbelin

    Returns:
        tp_agg, a pd.DataFrame with a row for each day of the week/machine combination, with information on the
        starting and finishing time for the MR machine, along with a date range for which these times are applicable
        for. A column containing the total number of minutes in the day is included as well.
    """
    tp_df = terminplanner_df.copy()
    tp_df['Termin'] = pd.to_datetime(tp_df['Termin'], format='%H:%M')
    tp_df['Terminbuch'] = tp_df['Terminbuch'].replace({'MR1': 1, 'MR2': 2})
    tp_df['Wochentag'] = tp_df['Wochentag'].replace({'MO': 'Monday',
                                                     'DI': 'Tuesday',
                                                     'MI': 'Wednesday',
                                                     'DO': 'Thursday',
                                                     'FR': 'Friday'
                                                     })
    tp_df['Dauer in dt'] = pd.to_timedelta(tp_df['Dauer in Min.'], unit='m')
    tp_df['terminende'] = tp_df['Termin'] + tp_df['Dauer in dt']
    tp_df['Termin'] = tp_df['Termin'].dt.time
    tp_df['terminende'] = tp_df['terminende'].dt.time
    tp_agg = tp_df.groupby(['Terminbuch', 'Wochentag', 'TERMINRASTER_NAME', 'gültig von', 'gültig bis']).agg({
        'Termin': 'min',
        'terminende': 'max',
        'Dauer in Min.': 'sum'
    }).reset_index()

    tp_agg.rename(columns={'Wochentag': 'day_of_week',
                           'Terminbuch': 'image_device_id',
                           'Termin': 'day_start_tp',
                           'terminende': 'day_end_tp',
                           'Dauer in Min.': 'day_length_tp',
                           'gültig von': 'applicable_from',
                           'gültig bis': 'applicable_to'}, inplace=True)
    return tp_agg


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

    idle_df['image_start'] = pd.to_datetime(idle_df['image_start'], errors='coerce')
    idle_df['image_end'] = pd.to_datetime(idle_df['image_end'], errors='coerce')

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

    # Below to prevent error when attempting to save as Parquet dataset
    # appts_and_gaps['start'] = appts_and_gaps['start'].astype('datetime64[s]')
    # appts_and_gaps['end'] = appts_and_gaps['end'].astype('datetime64[s]')

    return appts_and_gaps
