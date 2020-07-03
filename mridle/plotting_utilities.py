import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import random
import altair as alt
import seaborn as sns
from copy import deepcopy
from typing import Any, Dict
from mridle.data_management import validate_against_dispo_data


# ==================================================================
# ====== ALTAIR FUNCTIONS ==========================================
# ==================================================================

PROJECT_START_DATE = '01/14/2019'
PROJECT_END_DATE = '02/14/2019'

COLOR_SCHEME = {
    'orange': '#EC9139',
    'teal': '#54B89C',
    'purple': '#9329A0',
    'dark blue': '#3C50AF',
    'light blue': '#4CA8EF',
    'pink': '#D8538C',
    'yellow': '#EAC559',
    'red': '#CF2D20',
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


def alt_plot_date_range_for_device(df: pd.DataFrame, device: str = 'MR1', start_date: str = PROJECT_START_DATE,
                                   end_date: str = PROJECT_END_DATE, color_map: Dict = DEFAULT_COLOR_MAP,
                                   highlight: Any = None, height_factor: int = 10) -> alt.Chart:
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
    if start_date:
        df_filtered = df_filtered[df_filtered['start_time'] >= start_date]
    if end_date:
        df_filtered = df_filtered[df_filtered['start_time'] < end_date]
    if df_filtered.shape[0] == 0:
        raise ValueError('No data found in that date range')

    if device not in df['EnteringOrganisationDeviceID'].unique():
        raise ValueError('Device {} not found in data set'.format(device))
    df_filtered = df_filtered[df_filtered['EnteringOrganisationDeviceID'] == device].copy()

    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    if highlight is not None:
        plot_color_map = update_color_map_with_highlight(highlight, plot_color_map)

    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    recommended_height = df_filtered['start_time'].dt.to_period('D').nunique() * height_factor

    return alt.Chart(df_filtered).mark_bar().encode(
        alt.Color('slot_type:N', scale=color_scale),
        y='monthdate(start_time):T',
        x='hoursminutes(start_time):T',
        x2='hoursminutes(end_time):T',
        tooltip='FillerOrderNo'
    ).configure_mark(
        opacity=0.5,
    ).properties(
        width=800,
        height=recommended_height,
        title=device
    )


def alt_plot_day_for_device(df: pd.DataFrame, date: str, device: str, highlight: Any = None) -> alt.Chart:
    """
    Helper function to plot jst one day for a device
    Args:
        df: a one-row-per-slot dataframe.
        device: the device to plot
        date: date to plot
        highlight: string or list of strings. Highlight one type of appointment, rendering the rest in grey.

    Returns: alt.Chart

    """
    start_date = pd.to_datetime(date)
    end_date = start_date + pd.Timedelta(days=1)
    return alt_plot_date_range_for_device(df, device, start_date, end_date, highlight=highlight)


def update_color_map_with_highlight(highlight: Any, color_map: Dict, color_scheme: Dict = COLOR_SCHEME) -> Dict:
    """
    Given a higlight, grey out non-highlighted entries in an altair color_map.

    Args:
        highlight: string or list of strings to highlight in the color_map.
        color_map: dict of plotted types to color names.
        color_scheme: dict of color names to hex values.

    Returns: revised color_map

    """
    if highlight is not None:
        for entry in color_map:
            if type(highlight) == list:
                if entry not in highlight:
                    color_map[entry] = color_scheme['grey']
            if type(highlight) == str:
                if entry != highlight:
                    color_map[entry] = color_scheme['grey']
    return color_map


def plot_example_day_against_dispo(slot_df: pd.DataFrame, dispo_df: pd.DataFrame, date, device='MR1',
                                   color_map=DETAILED_COLOR_MAP, stroke_map=STROKE_MAP, anonymize=True,
                                   height_factor=20):
    """
    Plot completed, inpatient, and no-show appointments for one device for a day.

    Args:
        df: a one-row-per-slot dataframe.
        device: the device to plot
        date: date to plot
        color_map: the colors to use for each appointment type.
        stroke_map: the colors to use for each NoShow_outcome type.
        anonymize: whether to anonymize the data by shifting it by a random amount.
        height_factor: multiplier for how tall to make the plot based on the number of days plotted

    Returns: alt.Chart

    """

    # prep dispo_df and slot_df and combine them
    dispo_df_copy = dispo_df.copy()
    dispo_df_copy['source'] = 'dispo'
    dispo_df_copy['end_time'] = dispo_df_copy['start_time'] + pd.to_timedelta(30, unit='minutes')
    dispo_df['EnteringOrganisationDeviceID'] = dispo_df['machine']

    slot_df_copy = slot_df.copy()
    slot_df_copy['patient_id'] = slot_df_copy['MRNCmpdId']
    slot_df_copy['source'] = 'extract'

    slot_val_compare_df = pd.concat([dispo_df_copy, slot_df_copy])

    # filter
    start_date = pd.to_datetime(date)
    end_date = start_date + pd.Timedelta(days=1)
    slot_val_compare_df = slot_val_compare_df[slot_val_compare_df['start_time'] >= start_date]
    slot_val_compare_df = slot_val_compare_df[slot_val_compare_df['start_time'] < end_date]
    if slot_val_compare_df.shape[0] == 0:
        raise ValueError('No data found in that date range')

    if device not in slot_val_compare_df['EnteringOrganisationDeviceID'].unique():
        raise ValueError('Device {} not found in data set'.format(device))
    slot_val_compare_df = slot_val_compare_df[slot_val_compare_df['EnteringOrganisationDeviceID'] == device].copy()

    title = device

    if anonymize:
        randomizer = pd.Timedelta(minutes=random.randint(-15, 15))
        slot_val_compare_df['start_time'] = slot_val_compare_df['start_time'] + randomizer
        slot_val_compare_df['end_time'] = slot_val_compare_df['end_time'] + randomizer
        title = 'Anonymized Appointment Data'

    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))
    stroke_scale = alt.Scale(domain=list(stroke_map.keys()), range=list(stroke_map.values()))

    recommended_height = len(DEFAULT_COLOR_MAP) * height_factor

    return alt.Chart(slot_val_compare_df).mark_bar(strokeWidth=3).encode(
        y=alt.Y('source:N', title='Source'),
        x=alt.X('hoursminutes(start_time):T', title='Time'),
        x2=alt.X2('hoursminutes(end_time):T'),
        color=alt.Color('slot_type_detailed:N', scale=color_scale, legend=alt.Legend(title='Slot Type (detailed)')),
        stroke=alt.Stroke('NoShow_outcome', scale=stroke_scale, legend=alt.Legend(title='Canceled Appts')),
        tooltip='patient_id',
    ).configure_mark(
        opacity=0.5,
    ).properties(
        width=800,
        height=recommended_height,
        title=title
    )


# ==================================================================
# ====== MATPLOTLB FUNCTIONS =======================================
# ==================================================================


def plot_a_day(df: pd.DataFrame, year: int, month: int, day: int, labels: bool = True, alpha: float = 0.5) -> None:
    """
    Plot completed, inpatient, and no-show appointments across all devices for one day.

    Args:
        df: Row-per-appt-slot dataframe.
        year: Year of the day to plot.
        month: Month of the day to plot.
        day: Day number of the day to plot.
        labels: Whether to show appointment ids (FillerOrderNo).
        alpha: Transparency level for the plotted appointments.

    Returns: None

    """
    one_day = df[(df['start_time'].dt.year == year)
                 & (df['start_time'].dt.month == month)
                 & (df['start_time'].dt.day == day)].copy()

    # ======

    slot_type_color_map = {
        'show': 'tab:blue',
        'no-show': 'tab:red',
        'inpatient': 'tab:gray',
    }

    label_col = 'MRNCmpdId'
    no_phi_label_col = 'FillerOrderNo'
    if label_col not in df.columns:
        label_col = no_phi_label_col

    device_row_height = 10
    default_duration = pd.Timedelta(minutes=30)

    # enter default end times for no shows
    one_day['duration'] = np.where(one_day['slot_type'] == 'no-show',
                                   default_duration,
                                   one_day['end_time'] - one_day['start_time']
                                   )
    one_day['duration'] = pd.to_timedelta(one_day['duration'])

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    yticks = []
    ytick_labels = []

    # make a row for each device...
    devices = one_day['EnteringOrganisationDeviceID'].unique()
    devices = [x for x in devices if not pd.isnull(x)]
    for device_index, device in enumerate(devices):
        device_height = device_index * device_row_height
        yticks.append(device_height + device_row_height / 2)
        ytick_labels.append(device)

        # plot each type of appointment slot in a different color...
        for slot_type in one_day['slot_type'].unique():
            plot_data_subset = one_day[(one_day['EnteringOrganisationDeviceID'] == device) &
                                       (one_day['slot_type'] == slot_type)]
            plot_data_subset_tuples = [(row['start_time'], row['duration'])
                                       for i, row in plot_data_subset.iterrows()
                                       ]
            ax.broken_barh(plot_data_subset_tuples, (device_height, 9), facecolors=slot_type_color_map[slot_type],
                           edgecolor=slot_type_color_map[slot_type], alpha=alpha)

            # add PatID/FON labels
            if labels:
                fon_labels = [(row['start_time'],
                               row[label_col])
                              for i, row in plot_data_subset.iterrows()
                              ]
                for t in fon_labels:
                    ax.text(x=t[0] + default_duration / 2, y=device_height + device_row_height / 2, s=t[1], ha='center',
                            va='center',
                            rotation=40)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel('Device')

    hours = mdates.HourLocator()
    ax.xaxis.set_major_locator(hours)
    xfmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Time')
    fig.autofmt_xdate()
    ax.autoscale()

    # plt.title(pd.Timestamp(year=year, month=month, day=day))
    plt.title(one_day['start_time'].iloc[0].strftime('%d %B, %Y'))

    legend_elements = [matplotlib.patches.Patch(facecolor='blue', alpha=alpha, edgecolor='b',
                                                label='Ambulatory Appointments'),
                       matplotlib.patches.Patch(facecolor='red', alpha=alpha, edgecolor='r',
                                                label='Ambulatory No-Shows'),
                       matplotlib.patches.Patch(facecolor='grey', alpha=alpha, edgecolor='g',
                                                label='In-Patient Appointments')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()


def plot_a_day_for_device(df: pd.DataFrame, device: str, year: int, month: int, day: int, labels: bool = True,
                          alpha: float = 0.5, jitter: bool = True):
    """
    Plot completed, inpatient, and no-show appointments for one device for one day.

    Args:
        df: Row-per-appt-slot dataframe.
        device: Device to plot, as set in EnteringOrganisationDeviceID.
        year: Year of the day to plot.
        month: Month of the day to plot.
        day: Day number of the day to plot.
        labels: Whether to show appointment ids (FillerOrderNo).
        alpha: Transparency level for the plotted appointments.
        jitter: Whether to jitter the appointment boxes for easier discerning of their boundaries.

    Returns: None

    """
    if device not in df['EnteringOrganisationDeviceID'].unique():
        raise ValueError('Device {} not found in data set'.format(device))

    one_day = df[(df['start_time'].dt.year == year)
                 & (df['start_time'].dt.month == month)
                 & (df['start_time'].dt.day == day)
                 & (df['EnteringOrganisationDeviceID'] == device)].copy()

    # ======

    slot_type_color_map = {
        'show': 'tab:blue',
        'no-show': 'tab:red',
        'inpatient': 'tab:gray',
    }

    label_col = 'MRNCmpdId'
    no_phi_label_col = 'FillerOrderNo'
    if label_col not in df.columns:
        label_col = no_phi_label_col

    row_height = 10
    default_duration = pd.Timedelta(minutes=30)

    # enter default end times for no shows
    one_day['duration'] = np.where(one_day['slot_type'] == 'no-show',
                                   default_duration,
                                   one_day['end_time'] - one_day['start_time']
                                   )
    one_day['duration'] = pd.to_timedelta(one_day['duration'])

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    yticks = []
    ytick_labels = []

    # make a row for each device...
    for index, slot_type in enumerate(one_day['slot_type'].unique()):
        height = index * row_height
        yticks.append(height + row_height / 2)
        ytick_labels.append(slot_type)

        plot_data_subset = one_day[(one_day['EnteringOrganisationDeviceID'] == device) &
                                   (one_day['slot_type'] == slot_type)]
        for i, row in plot_data_subset.iterrows():
            if jitter:
                display_height = height + random.uniform(-0.5, 0.5)
            ax.broken_barh([(row['start_time'], row['duration'])], (display_height, row_height - 1),
                           facecolors=slot_type_color_map[slot_type],
                           edgecolor=slot_type_color_map[slot_type], alpha=alpha)

            # add PatID/FON labels
            if labels:
                ax.text(x=row['start_time'] + default_duration / 2, y=display_height + row_height / 2,
                        s=row[label_col], ha='center', va='center', rotation=40)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    hours = mdates.HourLocator()
    ax.xaxis.set_major_locator(hours)
    xfmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
    ax.autoscale()

    # plt.title(pd.Timestamp(year=year, month=month, day=day))
    plt.title("{} - {}".format(device, one_day['start_time'].iloc[0].strftime('%d %B, %Y')))
    plt.show()


def plot_pos_class_freq_per_x(input_df: pd.DataFrame, var_x: str, var2freq: str, var_y: str):
    '''
    The objective of this function is to:
    (1) use the original feature set to generate a dataframe with three columns to get insights
    into the dataset: (a) var_x which is the variable of interest (b) Frequencies of var_y per
    value of var_x (c) Probability of var_y per value of var_x and
    (2) plot such dataframe

    Args:
        input_df: dataframe containing features for modeling
        var_x: variable of interest,e.g., 'historic_no_show_cnt'
        var2freq: frequency of var_y per value of var_x
        var_y: probability of var_y per value of var_x

    Returns: None
    '''
    # First make a crosstabulation - var2freq / var_x
    table_frequencies = pd.crosstab(input_df[var2freq], input_df[var_x], margins=True)

    # df_frequencies contains all the accumulated frequencies per value of X
    df_frequencies = table_frequencies.iloc[2]

    print(df_frequencies)
    df_frequencies = df_frequencies.reset_index()

    # New crosstabulation, now to get normalized values
    table_normalized = pd.crosstab(input_df[var2freq], input_df[var_x], margins=False, normalize='columns')

    # Extract the row with normalized probabilities
    output_df = table_normalized.iloc[1]
    output_df = output_df.reset_index()
    output_df = output_df.rename(columns={1: var_y})

    # Now add the frequencies per var_x in the new dataframe
    output_df['frequencies'] = df_frequencies['All']

    sns.set(style='white')
    sns.relplot(x=var_x, y=var_y, alpha=0.8, size='frequencies', sizes=(40, 400), data=output_df)


def plot_validation_experiment(df_ratio: pd.DataFrame) -> alt.Chart:
    """
    Plots variable in a scatter column plot per year. Variable is the ratio between official value and extract value
    obtained in pre-processing step.

    Args:
        df_ratio: dataframe with ratio calculated between dispo and extract

    Returns: plot

    """

    df_ratio["date"] = pd.to_datetime(df_ratio["date"])
    source = df_ratio

    stripplot = alt.Chart(source, width=40).mark_circle(size=50).encode(
        x=alt.X(
            'jitter:Q',
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y('ratio:Q'),
        color=alt.Color('year:N', legend=None),
        column=alt.Column(
            'year:N',
            header=alt.Header(
                labelAngle=-90,
                titleOrient='top',
                labelOrient='bottom',
                labelAlign='right',
                labelPadding=3,
            ),
        ),
    ).transform_calculate(
        # Generate Gaussian jitter with a Box-Muller transform
        jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )

    return stripplot


def plot_dispo_extract_outside_overlap(dispo_data: pd.DataFrame, slot_df: pd.DataFrame, slot_type_detailed: str):
    """
    Generates a scatter plot where evey point is represented by the (x, y) pair,
    x being the # of patients in the dispo_df that are not in the extract and
    y being the # of patients in the extract that are not in the dispo_df.

    Args:
        dispo_data: Dataframe with appointment data.
        slot_df: Dataframe with appointment data from extract.

    Returns: scatter plot explained above.
    """
    df = pd.DataFrame(columns=['year', 'dispo_not_extract', 'extract_not_dispo'])
    for date_elem in dispo_data.date.dt.date.unique():
        print('\nCurrent date: {}'.format(date_elem))
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # Identify how many appointments of a given 'type' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   slot_type_detailed)

        in_dispo_not_slot_df = len(dispo_patids.difference(slot_df_patids))
        in_slot_df_not_dispo = len(slot_df_patids.difference(dispo_patids))
        df = df.append({'year': date_elem.year, 'dispo_not_extract': in_dispo_not_slot_df,
                        'extract_not_dispo': in_slot_df_not_dispo}, ignore_index=True)

    df['year'] = df.year.astype(str)

    plot = alt.Chart(df).mark_point(size=60).encode(
        alt.X('dispo_not_extract', scale=alt.Scale(domain=(-1, 10), clamp=False)),
        alt.Y('extract_not_dispo', scale=alt.Scale(domain=(-1, 10), clamp=False)),
        color='year').interactive()

    return plot
