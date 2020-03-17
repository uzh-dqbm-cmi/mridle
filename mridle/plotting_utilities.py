import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import random


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

    slot_status_color_map = {
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
    one_day['duration'] = np.where(one_day['slot_status'] == 'no-show',
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
        for slot_status in one_day['slot_status'].unique():
            plot_data_subset = one_day[(one_day['EnteringOrganisationDeviceID'] == device) &
                                       (one_day['slot_status'] == slot_status)]
            plot_data_subset_tuples = [(row['start_time'], row['duration'])
                                       for i, row in plot_data_subset.iterrows()
                                       ]
            ax.broken_barh(plot_data_subset_tuples, (device_height, 9), facecolors=slot_status_color_map[slot_status],
                           edgecolor=slot_status_color_map[slot_status], alpha=alpha)

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

    slot_status_color_map = {
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
    one_day['duration'] = np.where(one_day['slot_status'] == 'no-show',
                                   default_duration,
                                   one_day['end_time'] - one_day['start_time']
                                   )
    one_day['duration'] = pd.to_timedelta(one_day['duration'])

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    yticks = []
    ytick_labels = []

    # make a row for each device...
    for index, slot_status in enumerate(one_day['slot_status'].unique()):
        height = index * row_height
        yticks.append(height + row_height / 2)
        ytick_labels.append(slot_status)

        plot_data_subset = one_day[(one_day['EnteringOrganisationDeviceID'] == device) &
                                   (one_day['slot_status'] == slot_status)]
        plot_data_subset_tuples = [(row['start_time'], row['duration'])
                                   for i, row in plot_data_subset.iterrows()
                                   ]
        for i, row in plot_data_subset.iterrows():
            if jitter:
                display_height = height + random.uniform(-0.5, 0.5)
            ax.broken_barh([(row['start_time'], row['duration'])], (display_height, row_height - 1),
                           facecolors=slot_status_color_map[slot_status],
                           edgecolor=slot_status_color_map[slot_status], alpha=alpha)

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
