import altair as alt
import altair_saver
import argparse
from copy import deepcopy
import os
import pandas as pd
import random
import sqlite3
import mridle
from datatc import DataManager
from mridle.plotting_utilities import DEFAULT_COLOR_MAP, DETAILED_COLOR_MAP, OUTCOME_STROKE_MAP


def plot_example_day(df: pd.DataFrame, date, device='MR1', color_map=DETAILED_COLOR_MAP, stroke_map=OUTCOME_STROKE_MAP,
                     anonymize=True, height_factor=20):
    """
    Plot completed, inpatient, and no-show appointments for one device for a day.

    Args:
        df: a one-row-per-slot dataframe.
        device: the device to plot
        date: date to plot
        color_map: the colors to use for each appointment type.
        stroke_map: the colors to use for each slot_outcome type.
        anonymize: whether to anonymize the data by shifting it by a random amount.
        height_factor: multiplier for how tall to make the plot based on the number of days plotted

    Returns: alt.Chart

    """
    df_filtered = df.copy()
    start_date = pd.to_datetime(date)
    end_date = start_date + pd.Timedelta(days=1)
    df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    if device not in df['EnteringOrganisationDeviceID'].unique():
        raise ValueError('Device {} not found in data set'.format(device))
    df_filtered = df_filtered[df_filtered['EnteringOrganisationDeviceID'] == device].copy()

    title = device

    if anonymize:
        randomizer = pd.Timedelta(minutes=random.randint(-15, 15))
        df_filtered['start_time'] = df_filtered['start_time'] + randomizer
        df_filtered['end_time'] = df_filtered['end_time'] + randomizer
        title = 'Anonymized Appointment Data'

    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))
    stroke_scale = alt.Scale(domain=list(stroke_map.keys()), range=list(stroke_map.values()))

    recommended_height = len(DEFAULT_COLOR_MAP) * height_factor

    return alt.Chart(df_filtered).mark_bar(strokeWidth=3).encode(
        y=alt.Y('slot_type:N', title='Slot Type'),
        x=alt.X('hoursminutes(start_time):T', title='Time'),
        x2=alt.X2('hoursminutes(end_time):T'),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        stroke=alt.Stroke('slot_outcome', scale=stroke_scale, legend=alt.Legend(title='Canceled Appts')),
        tooltip='FillerOrderNo',
    ).properties(
        width=800,
        height=recommended_height,
        title=title
    )


def plot_appt_types_over_time(df: pd.DataFrame, start_date, end_date, color_map=DETAILED_COLOR_MAP):
    df_filtered = df.copy()

    # create week column for plotting
    df_filtered['week'] = df_filtered['start_time'] - pd.to_timedelta(df_filtered['start_time'].dt.dayofweek, unit='d')
    df_filtered['week'] = pd.to_datetime(df_filtered['week'].dt.date)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # make sure the end date is at a week boundary
    end_date = end_date - pd.to_timedelta(end_date.dayofweek, unit='d')
    df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    return alt.Chart(df_filtered).mark_line(strokeWidth=3).encode(
        y=alt.Y('count(FillerOrderNo):N', title='Appt Count'),
        x=alt.X('yearmonthdate(week):T', title='Week', axis=alt.Axis(format='%b %d')),
        order=alt.Order("monthdate(week)"),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        tooltip='monthdate(week)',
    ).properties(
        width=475,
        height=250,
        # title=title
    )


def dayofweek_name(x):
    days_of_week = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    return days_of_week[x]


def plot_appt_types_by_day_of_week(df: pd.DataFrame, start_date, end_date, color_map=DETAILED_COLOR_MAP):
    df_filtered = df.copy()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    df_filtered['weekday'] = df_filtered['start_time'].dt.dayofweek
    day_of_week_pivot = pd.pivot_table(df_filtered, index='slot_type_detailed', columns='weekday',
                                       values='FillerOrderNo', aggfunc='count')
    day_of_week_pivot_percent = day_of_week_pivot / day_of_week_pivot.sum()
    melted = pd.melt(day_of_week_pivot_percent.reset_index(), id_vars='slot_type_detailed', value_name='percent')
    melted_filtered = melted[melted['weekday'] < 5].copy()
    melted_filtered['weekday'] = melted_filtered['weekday'].apply(dayofweek_name)

    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    return alt.Chart(melted_filtered).mark_line(strokeWidth=3).encode(
        x=alt.X('weekday', title='Day of the Week', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']),
        y=alt.Y('percent', title='Percent of Total Appointments', axis=alt.Axis(format='%')),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        tooltip='percent',
    ).properties(
        width=250,
        height=250,
        # title=title
    )


def plot_dave_b(slot_df: pd.DataFrame, dicom_times_df: pd.DataFrame, example_date: str, start_date: str, end_date: str,
                anonymize: bool = True) -> alt.Chart:
    """
    Create the Dave B figure, which consists of three subplots:
    - an example day (uses slot_w_dicom_df to show actual times, rather than scheduled)
    - aggregate counts of appointments over time
    - patterns in slot frequencies by day of week

    Args:
        slot_df: slot_df
        dicom_times_df: Start and end times of appointment slots as determined by the dicom images.
        example_date: date to represent in the example day subplot. If none, a random date between start_date and
         end_date is chosen.
        start_date: start date for the 2 aggregate plots
        end_date: end date for the 2 aggregate plots
        anonymize: Whether to anonymize the example_day subplot

    Returns: altair plot
    """
    slot_w_dicom_df = mridle.data_management.integrate_dicom_data(slot_df, dicom_times_df)

    if example_date is None:
        # choose a random date
        random_row = slot_w_dicom_df[(~slot_w_dicom_df['start_time'].isna())
                                     & (slot_w_dicom_df['start_time'] > start_date)
                                     & (slot_w_dicom_df['end_time'] > end_date)
                                     ].sample(1)
        example_date = random_row['start_time'].dt.floor('d').iloc[0]

    example_day = plot_example_day(slot_w_dicom_df, example_date, anonymize=anonymize)
    daily_over_time = plot_appt_types_over_time(slot_df, start_date, end_date)
    day_of_week = plot_appt_types_by_day_of_week(slot_df, start_date, end_date)
    return (example_day & (daily_over_time | day_of_week)).configure_mark(opacity=0.75)


def plot_appt_len_vs_var(dicom_df: pd.DataFrame, variable: str, sort_order: list):
    return alt.Chart(dicom_df[[variable, 'appt_len_float', 'UniversalServiceName']]).mark_boxplot().encode(
        alt.X(variable, sort=sort_order),
        y='appt_len_float'
    ).properties(
        width=180,
        height=180
    ).facet(
        column='UniversalServiceName'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='mridle', help='The path to the data directory, or key to DataManager')
    parser.add_argument('--output_dir', default='mridle', help='The path of the location to save the figure to')
    parser.add_argument('--example_date', default=None, help='The date to plot in the example day subplot')
    parser.add_argument('--start_date', default='01/14/2019', help='The start date for the summary subplots')
    parser.add_argument('--end_date', default='05/01/2019', help='The end date for the summary subplots')
    args = parser.parse_args()

    dm = DataManager(args.data_dir)
    raw_df = dm['rdsc_extracts'].latest().select('xlsx').load()
    status_df = mridle.data_management.build_status_df(raw_df)
    slot_df = mridle.data_management.build_slot_df(status_df)

    dicom_db_path = dm['dicom_data'].latest().select('sqlite').path
    query_text = dm['dicom_data'].latest().select('image_times.sql').load()
    c = sqlite3.connect(dicom_db_path)
    dicom_times_df = pd.read_sql_query(query_text, c)
    dicom_times_df = mridle.data_management.format_dicom_times_df(dicom_times_df)

    chart = plot_dave_b(slot_df, dicom_times_df, example_date=args.example_date, start_date=args.start_date,
                        end_date=args.end_date, anonymize=True)
    altair_saver.save(chart, os.path.join(args.output_dir, 'dave_b_1.png'))


if __name__ == '__main__':
    main()
