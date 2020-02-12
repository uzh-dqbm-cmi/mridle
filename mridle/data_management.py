"""
All processing functions for the data transformation pipeline.

### Major Data Processing Steps ###

load_data():
 - reads raw file from filesystem and adds custom columns.
 - This data is in the format one-row-per-appt-status-change

build_slot_df():
 - returns data in the form one-row-per-appointment-slot (now show or completed appointment)

"""

import pandas as pd
import numpy as np
from typing import Dict, List


STATUS_MAP = {
        'p': 'requested',
        't': 'scheduled',
        'a': 'registered',
        'b': 'started',
        'u': 'examined',
        'd': 'dictated',
        's': 'cancelled',
        'f': 'verified',
        'g': 'deleted',
        'w': 'waiting',
    }

SHOW_COLS = ['FillerOrderNo', 'date', 'was_status', 'was_sched_for', 'now_status', 'now_sched_for', 'NoShow',
             'was_sched_for_date', 'now_sched_for_date']

RELEVANT_MACHINES = ['MR1', 'MR2', 'MRDL']
SERVICE_NAMES_TO_EXCLUDE = ['Zweitbefundung MR']


def build_status_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load data from the file system.

    Args:
        raw_df: raw data file of RIS export.

    Returns: Dataframe with one row per appointment status change.

    """
    df = raw_df.copy()
    df = convert_DtTm_cols(df)
    df = restrict_to_relevant_machines(df, RELEVANT_MACHINES)
    df = exclude_irrelevant_service_names(df, SERVICE_NAMES_TO_EXCLUDE)
    df = add_custom_status_change_cols(df)
    return df


def convert_DtTm_cols(df: pd.DataFrame, known_datetime_cols: List[str] = None) -> pd.DataFrame:
    """
    Convert columns to pd.datetime format. Automatically converts columns with DtTm in the name, as well as columns
    passed in known_datetime_cols.

    Args:
        df: Dataframe apply to type conversions.
        known_datetime_cols: Columns to convert.

    Returns: Dataframe with datetime type conversions.

    """
    # if there are other columns that need to be coverted to datetime format, add them here
    if known_datetime_cols is None:
        known_datetime_cols = []

    time_cols = [col for col in df.columns if 'DtTm' in col]
    time_cols.extend([col for col in known_datetime_cols if col in df.columns])
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
    return df


def restrict_to_relevant_machines(df: pd.DataFrame, machines: List[str]) -> pd.DataFrame:
    df_machine_subset = df[df['EnteringOrganisationDeviceID'].isin(machines)].copy()
    return df_machine_subset


def exclude_irrelevant_service_names(df: pd.DataFrame, service_names_to_exclude: List[str]) -> pd.DataFrame:
    df_machine_subset = df[df['UniversalServiceName'].isin(service_names_to_exclude) == False].copy()
    return df_machine_subset


def add_custom_status_change_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom columns to row-per-status-change df, most notably shifts for previous status and previous scheduled time.

    Args:
        df: row-per-status-change df

    Returns:row-per-status-change df with more columns.

    """
    df.sort_values(['FillerOrderNo', 'History_MessageDtTm'])
    df['date'] = df['History_MessageDtTm']
    df['prev_status'] = df.groupby('FillerOrderNo')['History_OrderStatus'].shift(1)
    df['was_status'] = df['prev_status'].apply(lambda x: get_status_text(x))
    df['was_sched_for_date'] = df.groupby('FillerOrderNo')['History_ObsStartPlanDtTm'].shift(1)
    df['was_sched_for'] = (df['was_sched_for_date'] - df['History_MessageDtTm']).apply(lambda x: x.days)
    df['now_status'] = df['History_OrderStatus'].apply(lambda x: get_status_text(x))
    df['now_sched_for'] = (df['History_ObsStartPlanDtTm'] - df['History_MessageDtTm']).apply(lambda x: x.days)
    df['now_sched_for_date'] = df['History_ObsStartPlanDtTm']
    df['NoShow'] = df.apply(find_no_shows, axis=1)
    return df


def get_status_text(status_code: str) -> str:
    """Convert status code p/t/u/etc to its plain English name."""

    if status_code is None or pd.isnull(status_code):
        return None
    elif status_code in STATUS_MAP:
        return STATUS_MAP[status_code]
    else:
        return 'unknown: {}'.format(status_code)


def find_no_shows(row: pd.DataFrame) -> bool:
    """
    Algorithm for identifying no-show appointments.

    Args:
        row: A row from a database, as generated by using df.apply(axis=1).

    Returns: True/False whether the status change indicates a no-show.

    """
    threshold = 2
    ok_was_status_changes = ['requested']
    no_show_now_status_changes = ['scheduled']
    relevant_columns = ['date', 'was_sched_for_date', 'was_status', 'now_status']
    for col in relevant_columns:
        if pd.isnull(row[col]):
            return False
    if row['PatientClass'] == 'ambulant' \
        and row['was_sched_for_date'] - row['date'] < pd.Timedelta(days=threshold) \
        and row['now_status'] in no_show_now_status_changes \
        and row['was_status'] not in ok_was_status_changes:
        return True
    return False


def build_slot_df(status_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert row-per-status-change dataframe to row-per-appointment-slot dataframe.

    Args:
        status_change_df: row-per-status-change dataframe.

    Returns: row-per-appointment-slot dataframe.
    """
    completed_appts = one_line_per_completed_appt(status_change_df)
    noshow_appts = one_line_per_no_show(status_change_df)
    one_per_slot = pd.concat([completed_appts, noshow_appts], sort=False)
    agg_dict = {
        'EnteringOrganisationDeviceID': 'min',
        'PatientClass': 'min',
        'UniversalServiceId': 'min',
        'UniversalServiceName': 'min',
    }
    one_per_slot_plus_details = add_column_details(status_change_df, one_per_slot, agg_dict)
    one_per_slot_plus_details['slot_status'] = np.where(one_per_slot_plus_details['PatientClass'] == 'stationär',
                                                        'inpatient',
                                                        one_per_slot_plus_details['slot_status'])
    return one_per_slot_plus_details


def one_line_per_completed_appt(status_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the start and end time of all completed appointments in a row-per-status-change dataframe.

    Args:
        status_change_df: row-per-status-change dataframe

    Returns: dataframe with columns ['FillerOrderNo', 'start_time', 'end_time', 'slot_status'].

    """
    completed_appts_start_times = status_change_df[(status_change_df['OrderStatus'] == 'u')
                                           & (status_change_df['now_status'] == 'started')
                                           & (status_change_df['UniversalServiceName'] != 'Fremduntersuchung MR')  # forieign
                                     ].groupby('FillerOrderNo').agg({'date': 'min'})
    completed_appts_start_times.columns = ['start_time']

    completed_appts_end_times = status_change_df[(status_change_df['OrderStatus'] == 'u')
                                   & (status_change_df['now_status'] == 'examined')
                                   ].groupby('FillerOrderNo').agg({'date': 'max'})
    completed_appts_end_times.columns = ['end_time']

    completed_appts = pd.merge(completed_appts_start_times, completed_appts_end_times, left_index=True,
                               right_index=True)
    completed_appts['slot_status'] = 'show'
    completed_appts.reset_index(inplace=True)
    return completed_appts


def one_line_per_no_show(status_change_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the start and end time of all completed appointments in a row-per-status-change dataframe.

    Args:
        status_change_df: row-per-status-change dataframe

    Returns: dataframe with columns ['FillerOrderNo', 'start_time', 'slot_status'].

    """
    noshow_appts = status_change_df[status_change_df['NoShow']][['FillerOrderNo', 'was_sched_for_date']].copy()
    noshow_appts.columns = ['FillerOrderNo', 'start_time']
    noshow_appts['slot_status'] = 'no-show'
    return noshow_appts


def add_column_details(detail_df: pd.DataFrame, slot_df: pd.DataFrame, agg_dict: Dict) -> pd.DataFrame:
    """
    Adds columns to slot_df, as determined by agg_dict and extracted from detail_df.

    Args:
        detail_df: row-per-status-change dataframe
        slot_df: row-per-appointment-slot dataframe
        agg_dict: dictionary defining the columns to add to slot_df and their aggregation methods.
            To be used in df.groupby().agg()

    Returns: slot_df with more columns.

    """
    appt_details = detail_df.groupby('FillerOrderNo').agg(agg_dict)
    df_with_details = pd.merge(slot_df, appt_details, left_on='FillerOrderNo', right_index=True, how='left')
    return df_with_details
