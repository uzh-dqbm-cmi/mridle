# data_management.py from 64e72eb
# https://github.com/uzh-dqbm-cmi/mridle/pull/4
# 2020 June 4

import pandas as pd
from typing import Dict, List


STATUS_MAP = {
    'p': 'requested',
    't': 'scheduled',
    'a': 'registered',
    'b': 'started',
    'u': 'examined',
    'd': 'dictated',
    's': 'canceled',
    'f': 'verified',
    'g': 'deleted',
    'w': 'waiting',
}

SHOW_COLS = ['FillerOrderNo', 'date', 'was_status', 'was_sched_for', 'now_status', 'now_sched_for', 'NoShow',
             'was_sched_for_date', 'now_sched_for_date']

RELEVANT_MACHINES = ['MR1', 'MR2', 'MRDL']
SERVICE_NAMES_TO_EXCLUDE = ['Zweitbefundung MR', 'Fremduntersuchung MR']


def build_status_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the raw appointment status change dataset into a nicely formatted version.
    Args:
        raw_df: raw data file of RIS export.
    Returns: Dataframe with one row per appointment status change.
        The resulting dataframe has the columns (illustrative, not a complete list):
         - FillerOrderNo: int, appt id
         - date (MessageDtTm): datetime, the date and time of the status change
         - was_status: str, the status the appt changed from
         - now_status: str, the status the appt changed to
         - was_sched_for: int, number of days ahead the appt was sched for before status change relative to `date`
         - now_sched_for: int, number of days ahead the appt is sched for after status change relative to `date`
         - was_sched_for_date: datetime, the date the appt was sched for before status change
         - now_sched_for_date datetime, the date the appt is sched for after status change
         - patient_class_adj: patient class (adjusted) ['ambulant', 'inpatient']
         - NoShow: bool, [True, False]
         - NoShow_severity: str, ['hard', 'soft']
         - NoShow_outcome: str, ['rescheduled', 'canceled']
         - slot_type: str, ['no-show', 'show', 'inpatient']
         - slot_type_detailed: str, ['hard no-show', 'soft no-show', 'show', 'inpatient']
    """
    df = raw_df.copy()
    df = convert_DtTm_cols(df)
    df = restrict_to_relevant_machines(df, RELEVANT_MACHINES)
    df = exclude_irrelevant_service_names(df, SERVICE_NAMES_TO_EXCLUDE)
    df = add_custom_status_change_cols(df)
    df = format_patient_id_col(df)
    df['patient_class_adj'] = df['PatientClass'].apply(adjust_patient_class)
    df['NoShow'] = df.apply(find_no_shows, axis=1)
    df['NoShow_severity'] = df.apply(set_no_show_severity, axis=1)
    df['NoShow_outcome'] = df.apply(set_no_show_outcome, axis=1)
    df['slot_type'] = df.apply(set_slot_type, axis=1)
    df['slot_type_detailed'] = df.apply(set_slot_type_detailed, axis=1)
    return df


def build_slot_df(input_status_df: pd.DataFrame, agg_dict: Dict) -> pd.DataFrame:
    """
    Convert status_df into slot_df. Identify "show" and "no show" appointment slots from status_df,
    and synthesize into a single dataframe of all appointments that occurred or were supposed to occur (but no-show'ed).
    Args:
        input_status_df: row-per-status-change dataframe.
    Returns: row-per-appointment-slot dataframe.
        The resulting dataframe has the columns (illustrative, not a complete list):
         - FillerOrderNo: int, appt id
         - start_time: datetime, appt scheduled start time
         - end_time: datetime, appt scheduled end time
         - NoShow: bool, [True, False]
         - NoShow_outcome: str, ['rescheduled', 'canceled']
         - slot_type: str, ['no-show', 'show', 'inpatient']
         - slot_type_detailed: str, ['hard no-show', 'soft no-show', 'show', 'inpatient']
         - EnteringOrganisationDeviceID: str, device the appt was scheduled for
         - UniversalServiceName: str, the kind of appointment
         - MRNCmpdId (if available): int, patient id
    """
    status_df = input_status_df.copy()
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    status_df['start_time'] = status_df.apply(identify_start_times, axis=1)
    status_df['end_time'] = status_df.apply(identify_end_times, axis=1)
    status_df['end_time'] = status_df.groupby('FillerOrderNo')['end_time'].fillna(method='bfill')

    # this agg dict will be used for getting data about both show and no-show appt slots
    default_agg_dict = {
        'start_time': 'min',
        'end_time': 'min',
        'NoShow': 'min',
        'slot_type': 'min',
        'slot_type_detailed': 'min',
        'NoShow_outcome': 'min',
        'EnteringOrganisationDeviceID': 'last',  # 'min', CHANGED TO BE PARQUET COMPATIBLE
        'UniversalServiceName': 'last',  # 'min', CHANGED TO BE PARQUET COMPATIBLE
    }
    if agg_dict is None:  # ADDED TO BE BUILD_FEATURE_SET COMPATIBLE
        agg_dict = default_agg_dict
    if 'MRNCmpdId' in status_df.columns:
        agg_dict['MRNCmpdId'] = 'min'

    # there should be one show appt per FillerOrderNo
    show_slot_type_events = status_df[status_df['slot_type'].isin(['show', 'inpatient'])].copy()
    show_slot_df = show_slot_type_events.groupby(['FillerOrderNo']).agg(agg_dict).reset_index()

    # there may be multiple no-show appts per FillerOrderNo
    no_show_slot_type_events = status_df[status_df['NoShow']].copy()
    no_show_slot_df = no_show_slot_type_events.groupby(['FillerOrderNo', 'was_sched_for_date']).agg(
        agg_dict).reset_index()
    no_show_slot_df.drop('was_sched_for_date', axis=1, inplace=True)

    slot_df = pd.concat([show_slot_df, no_show_slot_df], sort=False)

    return slot_df


def find_no_shows(row: pd.DataFrame) -> bool:
    """
    Algorithm for identifying no-show appointments.
    No-shows must be:
     * ambulant
     * have a status change occur within <threshold> days of their scheduled appointment
     * have a status change between allowed statuses
     * appointment must not have been scheduled for midnight
    Args:
        row: A row from status_df, as generated by using status_df.apply(axis=1).
    Returns: True/False whether the status change indicates a no-show.
    """
    threshold = 2
    ok_was_status_changes = ['requested']
    no_show_now_status_changes = ['scheduled', 'canceled']
    relevant_columns = ['date', 'was_sched_for_date', 'was_status', 'now_status']
    for col in relevant_columns:
        if pd.isnull(row[col]):
            return False
    if row['patient_class_adj'] == 'ambulant' \
        and row['was_sched_for_date'] - row['date'] < pd.Timedelta(days=threshold) \
            and row['now_status'] in no_show_now_status_changes \
            and row['was_status'] not in ok_was_status_changes \
            and row['was_sched_for_date'].hour != 0:
        return True
    return False


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
    df_machine_subset = df[~df['UniversalServiceName'].isin(service_names_to_exclude)].copy()
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
    return df


def format_patient_id_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'MRNCmpdId' in df.columns:
        df['MRNCmpdId'] = df['MRNCmpdId'].str.replace('|USZ', '', regex=False)
    return df


def adjust_patient_class(original_patient_class: str) -> str:
    default_patient_class = 'ambulant'
    patient_class_map = {
        'ambulant': 'ambulant',
        'stationär': 'inpatient',
        'teilstationär': 'inpatient',
    }
    if pd.isnull(original_patient_class):
        return default_patient_class
    elif original_patient_class in patient_class_map.keys():
        return patient_class_map[original_patient_class]
    else:
        return 'unknown'


def set_no_show_severity(row: pd.DataFrame) -> str:
    if row['NoShow']:
        if row['date'] > row['was_sched_for_date']:
            return 'hard'
        else:
            return 'soft'


def set_no_show_outcome(row: pd.DataFrame) -> str:
    if row['NoShow']:
        if row['now_status'] == 'canceled':
            return 'canceled'
        else:
            return 'rescheduled'


def set_slot_type(row: pd.DataFrame) -> str:
    if row['NoShow']:
        return 'no-show'
    elif row['OrderStatus'] == 'u' and row['now_status'] == 'started':
        if row['patient_class_adj'] == 'ambulant':
            return 'show'
        elif row['patient_class_adj'] == 'inpatient':
            return 'inpatient'
    else:
        return None


def set_slot_type_detailed(row: pd.DataFrame) -> str:
    if row['NoShow']:
        return '{} no-show'.format(row['NoShow_severity'])
    elif row['OrderStatus'] == 'u' and row['now_status'] == 'started':
        if row['patient_class_adj'] == 'ambulant':
            return 'show'
        elif row['patient_class_adj'] == 'inpatient':
            return 'inpatient'
    else:
        return None


def get_status_text(status_code: str) -> str:
    """Convert status code p/t/u/etc to its plain English name."""

    if status_code is None or pd.isnull(status_code):
        return None
    elif status_code in STATUS_MAP:
        return STATUS_MAP[status_code]
    else:
        return 'unknown: {}'.format(status_code)


def identify_start_times(row: pd.DataFrame) -> pd.datetime:
    """
    Identify start times of  appts. Could be used like this:
      status_df['start_time'] = status_df.apply(identify_end_times, axis=1)
    Args:
        row: row from status_df, as generated by using status_df.apply(axis=1).
    Returns: appt start datetime, or None if the row is not an appt starting event.
    """
    if row['NoShow']:
        return row['was_sched_for_date']
    elif row['now_status'] == 'started':
        return row['date']
    else:
        return None


def identify_end_times(row: pd.DataFrame) -> pd.datetime:
    """
    Identify end times of appts. Could be used like this:
      status_df['end_time'] = status_df.apply(identify_end_times, axis=1)
      status_df['end_time'] = status_df.groupby('FillerOrderNo')['end_time'].fillna(method='bfill')
    Args:
        row: row from status_df, as generated by using status_df.apply(axis=1).
    Returns: appt end datetime, or None if the row is not an appt ending event.
    """
    if row['NoShow']:
        return row['was_sched_for_date'] + pd.to_timedelta(30, unit='minutes')
    elif row['now_status'] == 'examined':
        return row['date']
    else:
        return None


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
