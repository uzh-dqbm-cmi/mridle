import altair as alt
import argparse
from copy import deepcopy
import pandas as pd
import random
import sqlite3
import mridle
from datatc import DataManager


COLOR_SCHEME = {
    'orange': '#EC9139',
    'teal': '#54B89C',
    'purple': '#9329A0',
    'dark blue': '#3C50AF',
    'light blue': '#4CA8EF',
    'pink': '#D8538C',
    'yellow': '#EAC559',
    'red': '#CF2D20',
    'light red': '#E7968F',
    'green': '#6FC53B',
    'grey': '#BCBCBC',
    'dark purple': '#681E72',
    'light purple': '#C184C9',
}

DEFAULT_COLOR_MAP = {
    'show': 'blue',
    'no-show': 'red',
    'inpatient': 'grey',
}

DETAILED_COLOR_MAP = {
    'show': 'blue',
    'soft no-show': 'orange',
    'hard no-show': 'red',
    'inpatient': 'grey',
}
STROKE_MAP = {
    #     'rescheduled': 'black',
    'canceled': 'black',
}


def alt_plot_day_for_device_by_status(df: pd.DataFrame, date, device='MR1', color_map=DETAILED_COLOR_MAP,
                                      stroke_map=STROKE_MAP, anonymize=True, height_factor=20):
    """
    Use Altair to plot completed, inpatient, and no-show appointments for one device for a time range.

    Args:
        df: a one-row-per-slot dataframe.
        device: the device to plot
        start_date: starting date for date range to plot
        end_date: ending date for date range to plot (open interval, does not include end date).
        color_map: the colors to use for eah appointment type.
        highlight: string or list of strings. Highlight one type of appointment, rendering the rest in grey.
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
        stroke=alt.Stroke('NoShow_outcome', scale=stroke_scale, legend=alt.Legend(title='Canceled Appts')),
        tooltip='FillerOrderNo',
    ).configure_mark(
        opacity=0.5,
    ).properties(
        width=800,
        height=recommended_height,
        title=title
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='mridle', help='The path to the data directory, or key to DataManager')
    parser.add_argument('--output_dir', default='mridle', help='The path of the location to save the figure to')
    parser.add_argument('--date', default=None, help='The date to plot')
    args = parser.parse_args()

    dm = DataManager(args.data_dir)
    raw_df = dm['rdsc_extracts'].latest().select('xlsx').load()
    status_df = mridle.data_management.build_status_df(raw_df)
    slot_df = mridle.data_management.build_slot_df(status_df)

    dicom_db_path = dm['dicom_data'].latest().select('sqlite').path
    query_text = dm['dicom_data'].latest().select('image_times.sql').load(data_interface_hint='txt')
    c = sqlite3.connect(dicom_db_path)
    dicom_times_df = pd.read_sql_query(query_text, c)
    dicom_times_df = mridle.data_management.format_dicom_times_df(dicom_times_df)
    slot_w_dicom_df = mridle.data_management.integrate_dicom_data(slot_df, dicom_times_df)

    if args.date is None:
        # choose a random date
        date = slot_w_dicom_df[~slot_w_dicom_df['start_time'].isna()].sample(1)['start_time'].dt.floor('d').iloc[0]
    else:
        date = args.date

    chart = alt_plot_day_for_device_by_status(slot_w_dicom_df, date=date, anonymize=True)
    chart.save('dave_b_1.png')


if __name__ == '__main__':
    main()
