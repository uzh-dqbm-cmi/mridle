import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Union
from mridle.utilities import data_processing


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


def prep_raw_df_for_parquet(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all dataframe columns to the appropriate data type. By default, the raw data is read in as mostly mixed-type
     'object' type columns, which Parquet does not accept. Most of these columns should instead be type 'category'
      (which also makes for a significantly lower memory footprint) or 'string'.

    Args:
        raw_df: raw data frame with many 'object' type columns.

    Returns: Dataframe with only int, datetime, category, and string data types.
    """
    date_cols = [
        'DateOfBirth',
    ]
    drop_cols = [
        'PlacerOrderNo',
        'StationTelefon',
    ]
    str_category_cols = [
        'History_OrderStatus',
        'OrderStatus',
        'PatientClass',
        'SAP FallArt',
        'Klasse',
        'InstituteID',
        'InstituteDivisionID',
        'EnteringOrganisationDeviceID',
        'UniversalServiceId',
        'UniversalServiceName',
        'DangerCode',
        'Sex',
        'Beruf',
        'Staatsangehoerigkeit',
        'WohnadrOrt',
        'WohnadrPLZ',
        'City',
        'Zivilstand',
        'Sprache',
        'MRNCmpdId',
        'StationName',
        'ApprovalStatusCode',
        'PerformProcedureID',
        'PerformProcedureName',
        'SourceFeedName',
    ]
    number_category_cols = [
        'Zip',

    ]

    string_cols = [
        'ReasonForStudy',
    ]

    df = data_processing.convert_DtTm_cols(raw_df, date_cols).copy()

    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    for col in str_category_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].astype('category')

    for col in number_category_cols:
        df[col] = df[col].apply(data_processing.nan_non_numbers)
        df[col] = df[col].astype('category')

    for col in string_cols:
        df[col] = df[col].astype(str)

    return df


def build_status_df(raw_df: pd.DataFrame,  exclude_patient_ids: List[str]) -> pd.DataFrame:
    """
    Clean up the raw appointment status change dataset into a nicely formatted version.

    Args:
        raw_df: raw data file of RIS export.
        exclude_patient_ids: List of patient ids that are test ids, and should be excluded.

    Returns: Dataframe with one row per appointment status change.
        The resulting dataframe has the columns (illustrative, not a complete list):
         - FillerOrderNo: int, appt id
         - date (History_MessageDtTm): datetime, the date and time of the status change
         - was_status: str, the status the appt changed from
         - now_status: str, the status the appt changed to
         - was_sched_for: int, number of days ahead the appt was sched for before status change relative to `date`
         - now_sched_for: int, number of days ahead the appt is sched for after status change relative to `date`
         - was_sched_for_date: datetime, the date the appt was sched for before status change
         - now_sched_for_date: datetime, the date the appt is sched for after status change
         - first_created_date: datetime, the earliest date the appointment was created
         - patient_class_adj: patient class (adjusted) ['ambulant', 'inpatient']
         - NoShow: bool, [True, False]
         - NoShow_severity: str, ['hard', 'soft']
         - slot_outcome: str, ['show', 'rescheduled', 'canceled']
         - slot_type: str, ['no-show', 'show', 'inpatient']
         - slot_type_detailed: str, ['hard no-show', 'soft no-show', 'show', 'inpatient']

    """
    df = raw_df.copy()
    df = data_processing.convert_DtTm_cols(df)
    df = restrict_to_relevant_machines(df, RELEVANT_MACHINES)
    df = exclude_irrelevant_service_names(df, SERVICE_NAMES_TO_EXCLUDE)
    df = add_custom_status_change_cols(df)
    df = format_patient_id_col(df)
    df = exclude_test_patient_ids(df, exclude_patient_ids)
    df = add_appt_first_created_date(df)
    df = add_final_scheduled_date(df)
    df['patient_class_adj'] = df['PatientClass'].apply(adjust_patient_class)
    df['NoShow'] = df.apply(find_no_shows, axis=1)
    df['NoShow_severity'] = df.apply(set_no_show_severity, axis=1)
    df['slot_type'] = df.apply(set_slot_type, axis=1)
    df['slot_outcome'] = df.apply(set_slot_outcome, axis=1)
    df['slot_type_detailed'] = df.apply(set_slot_type_detailed, axis=1)
    return df


def build_slot_df(input_status_df: pd.DataFrame, valid_date_range: List[str], agg_dict: Dict[str, str] = None,
                  include_id_cols: bool = True, build_future_slots: bool = False) -> pd.DataFrame:
    """
    Convert status_df into slot_df. Identify "show" and "no show" appointment slots from status_df,
    and synthesize into a single dataframe of all appointments that occurred or were supposed to occur (but no-show'ed).

    Args:
        input_status_df: row-per-status-change dataframe.
        valid_date_range: List of 2 strings defining the starting and ending date of the valid slot data period
         (status_df contains status change data outside the valid slot date range- these should not be made into slots).
        agg_dict: aggregation dict to pass to pd.DataFrame.agg() that specifies which columns from status_df to include
            in slot_df. It is recommended to aggregate by 'last' to use the latest value recorded for the slot. If no
            agg_dict is passed, the default will be used.
        include_id_cols: whether to include patient and appointment id columns in the resulting dataset.
        build_future_slots: whether we are building slot_df for appointments in the future to build dataset for
            predictions (and therefore no show/no-show type events yet), or we are building it 'normally' (with past
            data and show/no-show events) to train models with.

    Returns: row-per-appointment-slot dataframe.
        If no agg_dict is passed, the resulting dataframe has the following default columns:
         - FillerOrderNo: int, appt id
         - start_time: datetime, appt scheduled start time
         - end_time: datetime, appt scheduled end time
         - NoShow: bool, [True, False]
         - slot_outcome: str, ['show', 'rescheduled', 'canceled']
         - slot_type: str, ['no-show', 'show', 'inpatient']
         - slot_type_detailed: str, ['hard no-show', 'soft no-show', 'show', 'inpatient']
         - EnteringOrganisationDeviceID: str, device the appt was scheduled for
         - UniversalServiceName: str, the kind of appointment
         - MRNCmpdId (if available): str, patient id
    """
    default_agg_dict = {
        'patient_class_adj': 'last',
        'start_time': 'last',
        'end_time': 'last',
        'NoShow': 'min',
        'slot_outcome': 'last',
        'slot_type': 'last',
        'slot_type_detailed': 'last',
        'EnteringOrganisationDeviceID': 'last',
        'UniversalServiceName': 'last',
    }
    if agg_dict is None:
        agg_dict = default_agg_dict

    # start_time field is required for de-duping appts.
    # Add to agg_dict if it's not passed, but then remember to remove it later.
    if 'start_time' in agg_dict:
        start_time_requested_in_output = True
    else:
        start_time_requested_in_output = False
        agg_dict['start_time'] = 'min'

    status_df = input_status_df.copy()
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    status_df['start_time'] = status_df.apply(identify_start_times, axis=1)
    status_df['end_time'] = status_df.apply(identify_end_times, axis=1)
    status_df['end_time'] = status_df.groupby('FillerOrderNo')['end_time'].fillna(method='bfill')

    if build_future_slots:
        agg_dict['now_sched_for_date'] = 'last'
        future_slot_df = status_df.groupby(['FillerOrderNo', 'MRNCmpdId']).agg(agg_dict)
        future_slot_df['start_time'] = future_slot_df['now_sched_for_date']
        future_slot_df['end_time'] = future_slot_df['start_time'] + pd.to_timedelta(30, unit='minutes')
        future_slot_df.drop(columns=['now_sched_for_date'], inplace=True)
        if len(future_slot_df) > 0:
            # if there are 0 slots, the index column will be 'index', and reset_index will create an extra index col
            future_slot_df.reset_index(inplace=True)

        slot_df = future_slot_df.copy()
    else:
        # there should be one show appt per FillerOrderNo
        show_slot_type_events = status_df[status_df['slot_type'].isin(['show', 'inpatient'])].copy()
        show_slot_df = show_slot_type_events.groupby(['FillerOrderNo', 'MRNCmpdId']).agg(agg_dict)
        if len(show_slot_df) > 0:
            # if there are 0 shows, the index column will be 'index', and reset_index will create an extra index col
            show_slot_df.reset_index(inplace=True)

        # there may be multiple no-show appts per FillerOrderNo
        no_show_slot_type_events = status_df[status_df['NoShow']].copy()
        no_show_groupby_cols = ['FillerOrderNo', 'MRNCmpdId', 'was_sched_for_date']
        no_show_slot_df = no_show_slot_type_events.groupby(no_show_groupby_cols).agg(agg_dict)
        if len(no_show_slot_df) > 0:
            # if there are 0 no-shows, the index column will be 'index', and reset_index will create an extra index col
            no_show_slot_df.reset_index(inplace=True)
            no_show_slot_df.drop('was_sched_for_date', axis=1, inplace=True)

        slot_df = pd.concat([show_slot_df, no_show_slot_df], sort=False)

    # restrict to the valid date range
    valid_start_date, valid_end_date = valid_date_range
    day_after_last_valid_date = pd.to_datetime(valid_end_date) + pd.to_timedelta(1, 'days')
    slot_df = slot_df[slot_df['start_time'] >= valid_start_date]
    slot_df = slot_df[slot_df['start_time'] < day_after_last_valid_date]

    if len(slot_df) > 0:
        slot_df['FillerOrderNo'] = slot_df['FillerOrderNo'].astype(int)

        # filter out duplicate appointments for the same patient & time slot (weird dispo behavior)
        slot_df = data_processing.filter_duplicate_patient_time_slots(slot_df)

        if not include_id_cols:
            slot_df.drop('FillerOrderNo', axis=1, inplace=True)
            slot_df.drop('MRNCmpdId', axis=1, inplace=True)

        # remove start_time field if it wasn't requested in the passed agg_dict
        if start_time_requested_in_output:
            slot_df.sort_values('start_time', inplace=True)
        else:
            slot_df.drop('start_time', axis=1, inplace=True)

    slot_df.reset_index(drop=True, inplace=True)
    return slot_df


def find_no_shows(row: pd.DataFrame) -> bool:
    """
    Algorithm for identifying no-show appointments.

    No-shows must be:
     * ambulant
     * have a status change occur within <threshold> days of their scheduled appointment
     * have a status change between allowed statuses
     * appointment must not have been scheduled for midnight
     * appointment wasn't created on the same day that is is scheduled for

    Args:
        row: A row from status_df, as generated by using status_df.apply(axis=1).

    Returns: True/False whether the status change indicates a no-show.

    """
    threshold = 2
    ok_was_status_changes = ['requested']
    no_show_now_status_changes = ['scheduled', 'canceled']
    relevant_columns = ['date', 'was_sched_for_date', 'was_status', 'now_status', 'first_created_date']
    for col in relevant_columns:
        if pd.isnull(row[col]):
            return False
    if row['patient_class_adj'] == 'ambulant' \
        and np.busday_count(row['was_sched_for_date'], row['date']) < pd.Timedelta(days=threshold) \
            and row['now_status'] in no_show_now_status_changes \
            and row['was_status'] not in ok_was_status_changes \
            and row['was_sched_for_date'].hour != 0 \
            and row['first_created_date'].date() != row['was_sched_for_date'].date():
        return True
    return False


def restrict_to_relevant_machines(df: pd.DataFrame, machines: List[str]) -> pd.DataFrame:
    """Select a subset of input `df` where the `EnteringOrganisationDeviceID` column contains one of the values in
     input `machines`."""
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


def exclude_test_patient_ids(df: pd.DataFrame, exclude_patient_ids: List[str]) -> pd.DataFrame:
    """
    Exclude test patient ids. All patient ids with underscores are tests, and then there are also test patient ids that
     look normal but we have been told by the Dispo are test_ids.
    Args:
        df: Dataframe to exclude test patient_ids from.
        exclude_patient_ids: List of test patient ids.

    Returns: df without rows for test patient_ids.

    """
    df_result = df[~df['MRNCmpdId'].str.contains('_')].copy()
    df_result = df_result[~df_result['MRNCmpdId'].isin(exclude_patient_ids)].copy()
    return df_result


def add_appt_first_created_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "first_created_date" column to df, which is first date the appointment is created.
    Args:
        df: status_df

    Returns: status_df with additional first_created_date column

    """
    df.sort_values('date', inplace=True)
    final_scheduled_date_per_fon = df.groupby('FillerOrderNo').agg({'date': 'min'})
    final_scheduled_date_per_fon.columns = ['first_created_date']
    df = pd.merge(df, final_scheduled_date_per_fon, left_on='FillerOrderNo', right_index=True)
    return df


def add_final_scheduled_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "final_now_sched_for_date" column to df, which is the last now_sched_for_date value for each FillerOrderNo.
    Args:
        df: status_df

    Returns: status_df with additional final_now_sched_for_date column

    """
    df.sort_values('date', inplace=True)
    final_scheduled_date_per_fon = df.groupby('FillerOrderNo').agg({'now_sched_for_date': 'last'})
    final_scheduled_date_per_fon.columns = ['final_now_sched_for_date']
    df = pd.merge(df, final_scheduled_date_per_fon, left_on='FillerOrderNo', right_index=True)
    return df


def adjust_patient_class(original_patient_class: str) -> str:
    default_patient_class = 'ambulant'
    patient_class_map = {
        'ambulant': 'ambulant',
        'stationär': 'inpatient',
        'teilstationär': 'inpatient',
    }
    if pd.isnull(original_patient_class) or (type(original_patient_class) == str and original_patient_class == 'nan'):
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


def set_slot_outcome(row: pd.DataFrame) -> str:
    """
    Determine the outcome of the appointment slot: was it rescheduled, canceled, or did the patient show up?
    Appointment slots that are not moved into canceled status but are never rescheduled to a new time are labeled as
     canceled.

    Args:
        row: a row from status_df

    Returns: one of ['rescheduled', 'canceled', 'show', or 'inpatient']

    """
    if row['NoShow']:
        if row['now_status'] == 'canceled':
            return 'canceled'
        elif row['final_now_sched_for_date'] == row['was_sched_for_date']:
            return 'canceled'
        else:
            return 'rescheduled'
    elif row['slot_type'] == 'show' or row['slot_type'] == 'inpatient':
        return 'show'
    else:
        return None


def set_slot_type(row: pd.DataFrame) -> Union[str, None]:
    if row['NoShow']:
        return 'no-show'
    elif row['OrderStatus'] == 'u' and row['now_status'] == 'started':
        if row['patient_class_adj'] == 'ambulant':
            return 'show'
        elif row['patient_class_adj'] == 'inpatient':
            return 'inpatient'
    else:
        return None


def set_slot_type_detailed(row: pd.DataFrame) -> Union[str, None]:
    if row['NoShow']:
        return '{} no-show'.format(row['NoShow_severity'])
    elif row['OrderStatus'] == 'u' and row['now_status'] == 'started':
        if row['patient_class_adj'] == 'ambulant':
            return 'show'
        elif row['patient_class_adj'] == 'inpatient':
            return 'inpatient'
    else:
        return None


def get_status_text(status_code: str) -> Union[str, None]:
    """Convert status code p/t/u/etc to its plain English name."""

    if status_code is None or pd.isnull(status_code):
        return None
    elif status_code in STATUS_MAP:
        return STATUS_MAP[status_code]
    else:
        return 'unknown: {}'.format(status_code)


def identify_start_times(row: pd.DataFrame) -> Union[dt.datetime, None]:
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
        return row['was_sched_for_date']
    else:
        return None


def identify_end_times(row: pd.DataFrame) -> Union[dt.datetime, None]:
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
        return row['was_sched_for_date'] + pd.to_timedelta(30, unit='minutes')
    else:
        return None
