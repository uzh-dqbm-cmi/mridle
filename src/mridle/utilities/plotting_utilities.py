import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import random
import altair as alt
import seaborn as sns
from copy import deepcopy
from typing import Any, Dict, List, Set
from mridle.pipelines.data_engineering.dispo.nodes import jaccard_index


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

# determines the colors of the appointment slot boxes, when using slot_type
DEFAULT_COLOR_MAP = {
    'show': 'blue',
    'no-show': 'red',
    'inpatient': 'grey',
}

# determines the colors of the appointment slot boxes, when using slot_type_detailed
DETAILED_COLOR_MAP = {
    'show': 'blue',
    'soft no-show': 'orange',
    'hard no-show': 'red',
    'inpatient': 'grey',
}

OUTCOME_COLOR_MAP = {
    'show': 'blue',
    'rescheduled': 'orange',
    'canceled': 'red',
}

# determines whether the appointment slot box is outlined, for outlining no_show_outcome == 'canceled' appt slots
OUTCOME_STROKE_MAP = {
    'canceled': 'black',
}

NO_SHOW_TYPE_STROKE_MAP = {
    'hard no-show': 'black',
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
                                   color_map=OUTCOME_COLOR_MAP, stroke_map=NO_SHOW_TYPE_STROKE_MAP, anonymize=True,
                                   height_factor=20):
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

    # ensure dispo_df and slot_df have matching column names so they can be concat'ed
    dispo_df_copy = dispo_df.copy()
    dispo_df_copy['source'] = 'dispo'
    dispo_df_copy['end_time'] = dispo_df_copy['start_time'] + pd.to_timedelta(30, unit='minutes')
    dispo_df['EnteringOrganisationDeviceID'] = dispo_df['machine']

    slot_df_copy = slot_df.copy()
    slot_df_copy['patient_id'] = slot_df_copy['MRNCmpdId']
    slot_df_copy['source'] = 'extract'

    slot_val_compare_df = pd.concat([dispo_df_copy, slot_df_copy])

    # filter on date and device
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
        color=alt.Color('slot_outcome:N', scale=color_scale, legend=alt.Legend(title='Slot Outcome')),
        stroke=alt.Stroke('slot_type_detailed', scale=stroke_scale, legend=alt.Legend(title='No Show Type')),
        tooltip='patient_id',
    ).configure_mark(
        opacity=0.5,
    ).properties(
        width=800,
        height=recommended_height,
        title=title
    )


def plot_appt_len(slot_w_dicom_df: pd.DataFrame, exclude_herz: bool = False) -> alt.Chart:
    """
    Plot appointment length per appointment type. Appointment length is determined by DICOM start and end times, and
     appointment type by UniversalServiceName.

    Args:
        slot_w_dicom_df: result of `mridle.data_management.integrate_dicom_data(slot_df, dicom_times_df)`
        exclude_herz: Whether to exclude MR Herz from the plot (it's much bigger than the rest and makes the plot hard
          to read).

    """
    dicom_slot_df = slot_w_dicom_df[~slot_w_dicom_df['image_start'].isna()].copy()
    dicom_slot_df['appt_len'] = dicom_slot_df['image_end'] - dicom_slot_df['image_start']
    dicom_slot_df['appt_len_float'] = dicom_slot_df['appt_len'].dt.total_seconds() / 60
    plot_df = dicom_slot_df[['UniversalServiceName', 'appt_len_float']]
    plot_df = plot_df[plot_df['appt_len_float'] < 1000]

    if exclude_herz:
        plot_df = plot_df[plot_df['UniversalServiceName'] != 'MR Herz']

    appt_types = plot_df.groupby('UniversalServiceName').agg({'appt_len_float': 'mean'}).sort_values('appt_len_float',
                                                                                                     ascending=False)
    return alt.Chart(plot_df).mark_boxplot().encode(
        x=alt.X('UniversalServiceName', sort=list(appt_types.index)),
        y='appt_len_float:Q',
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


def plot_pos_class_freq_per_x(input_df: pd.DataFrame, var_x: str, var2freq: str, var_y: str, **kwargs: Any):
    """
    The objective of this function is to:
    (1) use the original feature set to generate a dataframe with three columns to get insights
    into the dataset: (a) var_x which is the variable of interest (b) Frequencies of var_y per
    value of var_x (c) Probability of var_y per value of var_x and
    (2) plot such dataframe

    Args:
        input_df: dataframe containing features for modeling
        var_x: variable of interest,e.g., 'no_show_before'
        var2freq: frequency of var_y per value of var_x
        var_y: probability of var_y per value of var_x
        **kwargs: user may add extra variables (a) xlim=[0, 100] (b) ylim=[0, 0.5]

    Returns: None
    """

    xlimits = kwargs.get('xlim', None)
    ylimits = kwargs.get('ylim', None)

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

    if xlimits is not None:
        plt.xlim(xlimits[0], xlimits[1])
    if ylimits is not None:
        plt.ylim(ylimits[0], ylimits[1])


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


def string_set(a_list):
    return set([str(i) for i in a_list])


def validate_against_dispo_data(dispo_data: pd.DataFrame, slot_df: pd.DataFrame, day: int, month: int, year: int,
                                slot_outcome: str, verbose: bool = False) -> Set[str]:

    """
    Identifies any appointment IDs that are in dispo_data or slot_df and not vice versa.
    Args:
        dispo_data: Dataframe with appointment data
        slot_df: Dataframe with appointment data from extract
        day: day numeric value
        month: month numeric value
        year: year numeric value
        slot_outcome: string with value ['show', 'rescheduled', 'canceled'].
            When `show` is selected, `inpatient` appointments are also included.
        verbose: whether to make prints during the comparison
    Returns:
        dispo_patids: set of strings with patient IDs from dispo
        slot_df_patids set of strings with patient IDs from extract
    """
    if slot_outcome not in ['show', 'rescheduled', 'canceled']:
        print('invalid type')
        return

    selected_dispo_rows = dispo_data[(dispo_data['date'].dt.day == day)
                                     & (dispo_data['date'].dt.month == month)
                                     & (dispo_data['date'].dt.year == year)
                                     & (dispo_data['slot_outcome'] == slot_outcome)
                                     ]
    selected_slot_df_rows = slot_df[(slot_df['start_time'].dt.day == day)
                                    & (slot_df['start_time'].dt.month == month)
                                    & (slot_df['start_time'].dt.year == year)
                                    & (slot_df['slot_outcome'] == slot_outcome)
                                    ]
    dispo_patids = string_set(list(selected_dispo_rows['patient_id'].unique()))
    slot_df_patids = string_set(list(selected_slot_df_rows['MRNCmpdId'].unique()))

    if verbose:
        print('{} Dispo Pat IDs: \n{}'.format(len(dispo_patids), dispo_patids))
        print('{} Slot_df Pat IDs: \n{}'.format(len(slot_df_patids), slot_df_patids))
        print()
        print('In Dispo but not in Slot_df: {}'.format(dispo_patids.difference(slot_df_patids)))
        print('In Slot_df but not in Dispo: {}'.format(slot_df_patids.difference(dispo_patids)))

    return dispo_patids, slot_df_patids


def plot_dispo_extract_slot_diffs(dispo_data: pd.DataFrame, slot_df: pd.DataFrame, slot_outcome: str):
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
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # Identify how many appointments of a given 'type' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   slot_outcome)

        in_dispo_not_slot_df = len(dispo_patids.difference(slot_df_patids))
        in_slot_df_not_dispo = len(slot_df_patids.difference(dispo_patids))
        df = df.append({'year': date_elem.year, 'dispo_not_extract': in_dispo_not_slot_df,
                        'extract_not_dispo': in_slot_df_not_dispo}, ignore_index=True)

    plot = alt.Chart(df).mark_point(size=60).encode(
        alt.X('dispo_not_extract', scale=alt.Scale(domain=(-1, 10), clamp=False)),
        alt.Y('extract_not_dispo', scale=alt.Scale(domain=(-1, 10), clamp=False)),
        color='year:O').interactive()

    return plot


def plot_scatter_bar_jaccard_per_type(dispo_data: pd.DataFrame, slot_df: pd.DataFrame, slot_outcome: str,
                                      color_map: Dict = OUTCOME_COLOR_MAP, highlight: Any = None):
    """
    Calculates the Jaccard Index per day, and plots the daily Jaccard values,
    split into subplots for each year.

    Args:
        color_map: the colors to use for eah appointment type.
        dispo_data: Dataframe with appointment data
        slot_df: Dataframe with appointment data from extract

    Returns: Scatter bar plot with each point representing the Jaccard index per day assessed.
    """
    # create color scale from color_map, modifying it based on highlight if appropriate
    plot_color_map = deepcopy(color_map)
    if highlight is not None:
        plot_color_map = update_color_map_with_highlight(highlight, plot_color_map)

    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    df = pd.DataFrame(columns=['year', 'jaccard', 'slot_type'])
    for date_elem in dispo_data.date.dt.date.unique():
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # Identify appointments for a given 'type' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   slot_outcome)

        jaccard = jaccard_index(dispo_patids, slot_df_patids)

        df = df.append({'year': date_elem.year, 'jaccard': jaccard, 'slot_type': slot_outcome}, ignore_index=True)

    stripplot = alt.Chart(df, width=40).mark_circle(size=50).encode(
        x=alt.X(
            'jitter:Q',
            title=None,
            axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
            scale=alt.Scale(),
        ),
        y=alt.Y('jaccard:Q'),
        color=alt.Color('slot_type:N', scale=color_scale),
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
    ).configure_facet(spacing=0).configure_view(stroke=None)

    return stripplot


def plot_scatter_dispo_extract_slot_cnt_for_type(dispo_data: pd.DataFrame, slot_df: pd.DataFrame, slot_outcome: str):
    """
    Generates a scatter plot where every point is represented by the (x, y) pair,
    x being the # of patients in the dispo_df,
    y being the # of patients in the extract_df,
    all of these for a given slot_outcome

    Args:
        dispo_data: dataframe with appointment data from the dispo nurse
        slot_df: dataframe with appointment data from extract
        slot_outcome: outcome of appointment slot

    Returns: plot
    """

    x = np.arange(-10, 50, 0.5)
    source = pd.DataFrame({
        'x': x,
        'y': x})

    plot_diagonal = alt.Chart(source).mark_circle(size=10).encode(
        x='x',
        y='y',
    )

    df = pd.DataFrame(columns=['year', 'appointments_in_dispo', 'appointments_in_extract'])
    for date_elem in dispo_data.date.dt.date.unique():
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # Identify how many 'shows' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   slot_outcome)

        df = df.append({'year': date_elem.year, 'appointments_in_dispo': len(dispo_patids),
                        'appointments_in_extract': len(slot_df_patids)}, ignore_index=True)

    plot_slot_cnt = alt.Chart(df).mark_point(size=60).encode(
        alt.X('appointments_in_dispo', scale=alt.Scale(domain=(-1, 40), clamp=False)),
        alt.Y('appointments_in_extract', scale=alt.Scale(domain=(-1, 40), clamp=False)),
        color='year:O').interactive()

    return plot_diagonal + plot_slot_cnt


def plot_scatter_dispo_extract_slot_cnt(dispo_data: pd.DataFrame, slot_df: pd.DataFrame, color_map=OUTCOME_COLOR_MAP):
    """
    Generates a scatter plot where every point is represented by the (x, y) pair,
    x being the # of patients in the dispo_df,
    y being the # of patients in the extract_df,
    all of these for a given slot_outcome

    Args:
        dispo_data: dataframe with appointment data from the dispo nurse
        slot_df: dataframe with appointment data from extract

    Returns: plot
    """

    x = np.arange(-10, 50, 0.5)
    source = pd.DataFrame({
        'x': x,
        'y': x})

    plot_diagonal = alt.Chart(source).mark_circle(size=10).encode(
        x='x',
        y='y',
    )

    df = pd.DataFrame(columns=['appointments_in_dispo', 'appointments_in_extract', 'slot_outcome'])
    for date_elem in dispo_data.date.dt.date.unique():
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # 'show'
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year, 'show')
        df = df.append({'appointments_in_dispo': len(dispo_patids), 'appointments_in_extract': len(slot_df_patids),
                        'slot_outcome': 'show'}, ignore_index=True)
        # 'rescheduled'
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   'rescheduled')
        df = df.append({'appointments_in_dispo': len(dispo_patids), 'appointments_in_extract': len(slot_df_patids),
                        'slot_outcome': 'rescheduled'}, ignore_index=True)
        # 'canceled'
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   'canceled')
        df = df.append({'appointments_in_dispo': len(dispo_patids), 'appointments_in_extract': len(slot_df_patids),
                        'slot_outcome': 'canceled'}, ignore_index=True)

    plot_color_map = deepcopy(color_map)
    color_scale = alt.Scale(domain=list(plot_color_map.keys()), range=list(plot_color_map.values()))

    plot_slot_cnt = alt.Chart(df).mark_circle(size=60).encode(
        alt.X('appointments_in_dispo', scale=alt.Scale(domain=(-1, 40), clamp=False)),
        alt.Y('appointments_in_extract', scale=alt.Scale(domain=(-1, 40), clamp=False)),
        color=alt.Color('slot_outcome', scale=color_scale)).interactive()

    return plot_diagonal + plot_slot_cnt


def plot_importances(var_importances: List, var_col_names: List):
    """
    Function that generates single importance plots for a single given model

    Args:
        var_importances is a list with single importances for a given model
        var_col_names is a list with variable names for a given model

    Returns: plot with importances for a single model
    """

    fig, ax = plt.subplots()
    width = 0.4  # the width of the bars
    ind = np.arange(len(var_importances))  # the x locations for the groups
    ax.barh(ind, var_importances, width, color='green')
    ax.set_yticks(ind + width / 10)
    ax.set_yticklabels(var_col_names, minor=False)
    plt.title('Feature importance in RandomForest Classifier')
    plt.xlabel('Relative importance')
    plt.ylabel('feature')
    plt.figure(figsize=(5, 5))
    fig.set_size_inches(6.5, 4.5, forward=True)
    plt.show()


def plot_importances_averages(var_importances_list: List[List[float]], var_col_names: List[str]):
    """
    Function that plots the importance averages for multiple models

    Args:
        var_importances_list: a 2 dimensional `list[i][j]` with all importance values for a
        given set of models, where `i` is the number of models and `j` is the number of features.
        val_col_names is a list with all variable names for the models.

    Returns: plot with average importances
    """
    average_list, std_list = [], []

    # Per model
    for i in range(0, len(var_importances_list[0])):
        values_to_avg = []
        # Per feature of interest
        for j in range(0, len(var_importances_list)):
            values_to_avg.append(var_importances_list[j][i])
        average = np.mean(np.array(values_to_avg))
        std = np.std(np.array(values_to_avg))
        average_list.append(average)
        std_list.append(std)

    fig, ax = plt.subplots()
    width = 0.4  # the width of the bars
    ind = np.arange(len(var_importances_list[0]))  # the x locations for the groups
    ax.barh(ind, average_list, width, color='green', xerr=std_list, capsize=5)
    ax.set_yticks(ind + width / 10)
    ax.set_yticklabels(var_col_names, minor=False)
    plt.xlim(0, 1)
    plt.title('Feature importance in RandomForest Classifier')
    plt.xlabel('Relative importance')
    plt.ylabel('feature')


def plot_importances_estimator(experiment: Any, cols_for_modeling: List):
    """
    Function generate plots importances for an estimator

    Args:
        experiment.model_runs is a set of runs for a given model trained
        in different folds within the dataset of interest.

    Returns: Importance plots
    """
    importance_score_list = []

    for model_index in range(len(experiment.model_runs)):
        current_model_name = 'Partition ' + str(model_index)
        modelrun_object = experiment.model_runs[current_model_name]

        clf_optimal = modelrun_object.model
        importances = clf_optimal.feature_importances_
        print('These are the -importance scores-: {}'.format(importances))
        col = cols_for_modeling

        plot_importances(importances, col)
        importance_score_list.append(importances)

    plot_importances_averages(importance_score_list, col)
