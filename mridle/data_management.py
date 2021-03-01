"""
All processing functions for the data transformation pipeline.

### Major Data Processing Steps ###

build_status_df():
 - reads raw file from filesystem and adds custom columns.
 - This data is in the format one-row-per-appt-status-change

build_slot_df():
 - returns data in the form one-row-per-appointment-slot (a slot is a no show or completed appointment)

"""

import datetime as dt
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Set, Union


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


# ========================================================================================
# === MAJOR TRANSFORMATION STEPS =========================================================
# ========================================================================================

def build_status_df(raw_df: pd.DataFrame,  exclude_patient_ids: List[str]) -> pd.DataFrame:
    """
    Clean up the raw appointment status change dataset into a nicely formatted version.

    Args:
        raw_df: raw data file of RIS export.
        exclude_patient_ids: List of patient ids that are test ids, and should be excluded.

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
         - slot_outcome: str, ['show', 'rescheduled', 'canceled']
         - slot_type: str, ['no-show', 'show', 'inpatient']
         - slot_type_detailed: str, ['hard no-show', 'soft no-show', 'show', 'inpatient']

    """
    df = raw_df.copy()
    df = convert_DtTm_cols(df)
    df = restrict_to_relevant_machines(df, RELEVANT_MACHINES)
    df = exclude_irrelevant_service_names(df, SERVICE_NAMES_TO_EXCLUDE)
    df = add_custom_status_change_cols(df)
    df = format_patient_id_col(df)
    df = exclude_test_patient_ids(df, exclude_patient_ids)
    df = add_final_scheduled_date(df)
    df['patient_class_adj'] = df['PatientClass'].apply(adjust_patient_class)
    df['NoShow'] = df.apply(find_no_shows, axis=1)
    df['NoShow_severity'] = df.apply(set_no_show_severity, axis=1)
    df['slot_type'] = df.apply(set_slot_type, axis=1)
    df['slot_outcome'] = df.apply(set_slot_outcome, axis=1)
    df['slot_type_detailed'] = df.apply(set_slot_type_detailed, axis=1)
    return df


def build_slot_df(input_status_df: pd.DataFrame, agg_dict: Dict[str, str] = None, include_id_cols: bool = True
                  ) -> pd.DataFrame:
    """
    Convert status_df into slot_df. Identify "show" and "no show" appointment slots from status_df,
    and synthesize into a single dataframe of all appointments that occurred or were supposed to occur (but no-show'ed).

    Args:
        input_status_df: row-per-status-change dataframe.
        agg_dict: aggregation dict to pass to pd.DataFrame.agg() that specifies which columns from status_df to include
            in slot_df. It is recommended to aggregate by 'last' to use the latest value recorded for the slot. If no
            agg_dict is passed, the default will be used.
        include_id_cols: whether to include patient and appointment id columns in the resulting dataset.

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

    # there should be one show appt per FillerOrderNo
    show_slot_type_events = status_df[status_df['slot_type'].isin(['show', 'inpatient'])].copy()
    show_slot_df = show_slot_type_events.groupby(['FillerOrderNo', 'MRNCmpdId']).agg(agg_dict)
    if len(show_slot_df) > 0:
        # if there are no shows, the index column will be 'index', and reset_index will create an extra index col
        show_slot_df.reset_index(inplace=True)

    # there may be multiple no-show appts per FillerOrderNo
    no_show_slot_type_events = status_df[status_df['NoShow']].copy()
    no_show_groupby_cols = ['FillerOrderNo', 'MRNCmpdId', 'was_sched_for_date']
    no_show_slot_df = no_show_slot_type_events.groupby(no_show_groupby_cols).agg(agg_dict)
    if len(no_show_slot_df) > 0:
        # if there are no no-shows, the index column will be 'index', and reset_index will create an extra index col
        no_show_slot_df.reset_index(inplace=True)
        no_show_slot_df.drop('was_sched_for_date', axis=1, inplace=True)

    slot_df = pd.concat([show_slot_df, no_show_slot_df], sort=False)
    if len(slot_df) > 0:
        slot_df['FillerOrderNo'] = slot_df['FillerOrderNo'].astype(int)

        # filter out duplicate appointments for the same patient & time slot (weird dispo behavior)
        slot_df = filter_duplicate_patient_time_slots(slot_df)

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
    slot_w_dicom_df = pd.merge(slot_df, dicom_times_df, how='left', left_on='FillerOrderNo', right_on='AccessionNumber')

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


# ========================================================================================
# === HELPER FUNCTIONS ===================================================================
# ========================================================================================

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def nan_non_numbers(x):
    if is_number(x):
        return x
    else:
        return np.nan


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

    df = convert_DtTm_cols(raw_df, date_cols).copy()

    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    for col in str_category_cols:
        # df[col] = df[col].astype(str)
        df[col] = df[col].astype('category')

    for col in number_category_cols:
        df[col] = df[col].apply(nan_non_numbers)
        df[col] = df[col].astype('category')

    for col in string_cols:
        df[col] = df[col].astype(str)

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
        try:
            df[col] = pd.to_datetime(df[col])
        except pd.errors.OutOfBoundsDatetime:
            df[col] = df[col].apply(fix_out_of_bounds_str_datetime)
            df[col] = pd.to_datetime(df[col])
    return df


def fix_out_of_bounds_str_datetime(val: str) -> str:
    """For string dates that have a year beyond 2100, replace the year with 2100.
    This is necessary because str -> datetime conversion fails for years beyond 2100."""
    if pd.isna(val):
        return np.nan
    match = re.match(r'.*([1-3][0-9]{3})', val)
    if match is None:
        return val
    year = match.group(1)
    if int(year) > 2100:
        val = val.replace(year, '2100')
        return val
    else:
        return val


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


def identify_start_times(row: pd.DataFrame) -> dt.datetime:
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


def identify_end_times(row: pd.DataFrame) -> dt.datetime:
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


def filter_duplicate_patient_time_slots(slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter duplicate patient-time slots. This is a scheduling behavior where two appointments are scheduled for the
     same patient for the same time, and right before the appt, one of the two is canceled. To prevent these
      appointments from being marked as no-shows, remove the no-show components of these duplicate appointments.

      Group slot_df by patient id and appt start time, sort by NoShow so shows (if present) are on top.
      Apply a cumcount, which marks the top entry with 0. Filter for all 0s, keeping only the single topmost entry
       for every patient-time-slot.

    Args:
        slot_df: slot_df, as generated by build_slot_df

    Returns: A subset of slot_df, where there is only one appointment per patient-time slot.

    """
    slot_df.sort_values(['MRNCmpdId', 'start_time', 'NoShow'], inplace=True)  # shows/NoShow == False will be on top
    slot_df['multi_slot'] = slot_df.groupby(['MRNCmpdId', 'start_time']).cumcount()
    first_slot_only = slot_df[slot_df['multi_slot'] == 0].copy()
    first_slot_only.drop(columns=['multi_slot'], axis=1, inplace=True)
    return first_slot_only


def build_dispo_exp_1_df(dispo_examples: List[Dict], exclude_patient_ids: List[str]) -> pd.DataFrame:
    """
    Convert the dispo data from validation experiment 1 into a dataframe and process it. Processing steps include:
    - formatting column data types
    - de-duping double show+canceled appointments

    Args:
        dispo_examples: Raw yaml list of dictionaries.
        exclude_patient_ids: List of patient ids that are test ids, and should be excluded.

    Returns: Dataframe of appointments collected in validation experiment 1.

    """
    dispo_slot_df = build_dispo_df(dispo_examples, exclude_patient_ids)

    # Ignore midnight appts with `slot_outcome == cancel` because these are not valid slots.
    # They are neither a `show` nor a `no show` (bc inpatient)
    dispo_slot_df['slot_outcome'] = np.where((dispo_slot_df['start_time'].dt.hour == 0) &
                                             (dispo_slot_df['slot_outcome'] == 'canceled'),
                                             '',
                                             dispo_slot_df['slot_outcome'])

    # use same de-duping function, create columns as necessary
    dispo_slot_df['MRNCmpdId'] = dispo_slot_df['patient_id']
    dispo_slot_df['NoShow'] = np.where(dispo_slot_df['slot_outcome'] == 'show', False, True)
    deduped_dispo_slot_df = filter_duplicate_patient_time_slots(dispo_slot_df)
    deduped_dispo_slot_df.drop(['MRNCmpdId'], axis=1, inplace=True)
    return deduped_dispo_slot_df


def build_dispo_exp_2_df(dispo_examples: List[Dict], exclude_patient_ids: List[str]) -> pd.DataFrame:
    """
        Convert the dispo data from validation experiment 2 into a dataframe and process it. Processing steps include:
        - formatting column data types
        - identifying rescheduled no shows from the sequence of dispo data points
        - de-duping double show+canceled appointments

        Args:
            dispo_examples: Raw yaml list of dictionaries.
            exclude_patient_ids: List of patient ids that are test ids, and should be excluded.

        Returns: Dataframe of appointments collected in validation experiment 1.

        """
    dispo_df = build_dispo_df(dispo_examples, exclude_patient_ids)
    dispo_slot_df = find_no_shows_from_dispo_exp_two(dispo_df)

    # use same de-duping function, create columns as necessary
    dispo_slot_df['MRNCmpdId'] = dispo_slot_df['patient_id']
    deduped_dispo_slot_df = filter_duplicate_patient_time_slots(dispo_slot_df)
    deduped_dispo_slot_df.drop(['MRNCmpdId'], axis=1, inplace=True)
    return deduped_dispo_slot_df


def build_dispo_df(dispo_examples: List[Dict], test_patient_ids: List[str]) -> pd.DataFrame:
    """
    Convert raw dispo data to dataframe and format the data types, namely dates and times.

    Args:
        dispo_examples: List of dictionaries containing appointment information collected at the Dispo.
        test_patient_ids: List of patient ids that are test ids, and should be excluded.

    Returns: Dataframe with datetime type conversions

    """
    dispo_df = pd.DataFrame(dispo_examples)
    dispo_df = dispo_df[~dispo_df['patient_id'].isin(test_patient_ids)].copy()
    dispo_df['patient_id'] = dispo_df['patient_id'].astype(int)
    dispo_df['start_time'] = pd.to_datetime(dispo_df['date'] + ' ' + dispo_df['start_time'], dayfirst=True)
    dispo_df['date'] = pd.to_datetime(dispo_df['date'], dayfirst=True)
    if 'date_recorded' in dispo_df.columns:
        dispo_df['date_recorded'] = pd.to_datetime(dispo_df['date_recorded'], dayfirst=True)
    return dispo_df


def find_no_shows_from_dispo_exp_two(dispo_e2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify no show events from the data collected in Validation Experiment 2.

    Args:
        dispo_e2_df: result of `build_dispo_df` on the dispo data collected for experiment 2.

    Returns: pd.DataFrame with one row per slot and 2 new columns: `NoShow` bool column and `slot_outcome` str column.

    """
    # calculate business days between date and recorded date
    # (np function requires datetime type)
    dispo_e2_df['date_dt'] = dispo_e2_df['date'].dt.date
    dispo_e2_df['date_recorded_dt'] = dispo_e2_df['date_recorded'].dt.date
    dispo_e2_df['date_diff'] = dispo_e2_df.apply(lambda x: np.busday_count(x['date_dt'], x['date_recorded_dt']), axis=1)
    dispo_e2_df.drop(columns=['date_dt', 'date_recorded_dt'], inplace=True)

    # Find the last time a slot was recorded before its start_time. Select all instances where the slot was recorded in
    # advance of the appt, and then pick the last occurance by applying a cumcount to the descending ordered list and
    # selecting the 0th row.
    before = dispo_e2_df[dispo_e2_df['date_diff'] < 0]
    before_pick_last = before.sort_values(['patient_id', 'date', 'date_recorded'], ascending=False)
    before_pick_last['rank'] = before_pick_last.groupby(['patient_id', 'date']).cumcount()
    before_last = before_pick_last[before_pick_last['rank'] == 0].copy()
    before_last.drop('rank', axis=1, inplace=True)

    # Select the rows where the slot was observed 1 business day after the slot date.
    after = dispo_e2_df[dispo_e2_df['date_diff'] == 1]

    one_day = pd.merge(before_last, after, how='outer', on=['patient_id', 'date', 'start_time'],
                       suffixes=('_before', '_after')
                       ).sort_values(['start_time'])

    def determine_dispo_no_show(last_status_before: str, first_status_after: str, last_status_date_diff: int,
                                start_time: pd.Timestamp) -> Union[bool, None]:
        """
        Determine whether a sequence of dispo data points collected in Validation Experiment 2 represents a no-show,
        based on the appointment's status before and after the appointment date, and the start time of the appointment.

        Args:
            last_status_before: The appointment status as of the last time the appointment was seen before the
             appointment date.
            first_status_after: The appointment status as of the first time the appointment was seen after the
             appointment date. If the appointment was rescheduled, then this will be None.
            last_status_date_diff: Number of days before the appt start date that the appt was seen.
            start_time: The time the appointment is scheduled to start at. If the start_time is midnight, the
             appointment is assumed to be an inpatient appointment, and automatically marked as False - not a no show.

        Returns: bool of whether the appointment is a no show, or None if the status cannot be determined.

        """
        if start_time.hour == 0:
            return False  # inpatient
        # only allow no-shows if the appt was recorded <=2 days in advance
        # i.e. exclude appts that were noted >2 days in advance but then moved before the 2 day window
        #  except do 3 days, because dispo data were collected at the end of the day, so something that was rescheduled
        #  on day -2 (a no-show!) was actually last captured manually on day -3
        #  except on day -3
        if last_status_date_diff < -2:
            return None
        elif last_status_before in ['ter', 'anm']:
            if first_status_after in ['bef', 'unt', 'schr']:
                return False  # show
            elif pd.isna(first_status_after):
                return True  # rescheduled no show
            elif first_status_after == 'ter':
                return True  # "to be rescheduled"?
        elif pd.isna(last_status_before) and first_status_after == 'bef':
            return False  # inpatient
        else:
            return None

    one_day['NoShow'] = one_day.apply(lambda x: determine_dispo_no_show(x['type_before'], x['type_after'],
                                                                        x['date_diff_before'], x['start_time']), axis=1)

    def determine_dispo_slot_outcome(no_show: bool, last_status_before: str, first_status_after: str,
                                     last_status_date_diff: int, start_time: pd.Timestamp) -> Union[str, None]:
        """
        Determine the slot_outcome of a sequence of dispo data points collected in Validation Experiment 2.

        Args:
            no_show: outcome of `determine_dispo_no_show`.
            last_status_before: The appointment status as of the last time the appointment was seen before the
             appointment date.
            first_status_after: The appointment status as of the first time the appointment was seen after the
             appointment date. If the appointment was rescheduled, then this will be None.
            last_status_date_diff: Number of days before the appt start date that the appt was seen.
            start_time: The time the appointment is scheduled to start at. If the start_time is midnight, the
             appointment is assumed to be an inpatient appointment, and automatically marked as False - not a no show.

        Returns: rescheduled, canceled, show, or None (not a slot)

        """
        if last_status_date_diff < -2:
            return None
        elif no_show:
            if last_status_before in ['ter', 'anm'] and (pd.isna(first_status_after)):
                return 'rescheduled'
            elif last_status_before in ['ter', 'anm'] and first_status_after == 'ter':
                return 'canceled'
            else:
                return None
        elif start_time.hour == 0:
            return None  # inpatient
        else:
            return 'show'

    one_day['slot_outcome'] = one_day.apply(lambda x: determine_dispo_slot_outcome(x['NoShow'], x['type_before'],
                                                                                   x['type_after'],
                                                                                   x['date_diff_before'],
                                                                                   x['start_time']), axis=1)

    return one_day


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


def validation_exp_confusion_matrix(dispo_df: pd.DataFrame, slot_df: pd.DataFrame, columns: List[str] = None
                                    ) -> pd.DataFrame:
    """
    Build a confusion matrix for each appointment found in dispo_df and how it is represented in slot_df.

    Args:
        dispo_df: result of `build_dispo_df`
        slot_df: result of `build_slot_df`
        columns: columns to keep in the confusion matrix (for exp 1, `['show', 'canceled']`,
         for exp 2, `['rescheduled', 'show']`)

    Returns: Confusion matrix data frame

    """
    d = dispo_df[['patient_id', 'start_time', 'slot_outcome']].copy()
    d['patient_id'] = d['patient_id'].astype(str)

    # filter slot_df to only the dates in dispo_df
    dispo_dates = dispo_df['date'].dt.date.unique()
    r = slot_df[slot_df['start_time'].dt.date.isin(dispo_dates)][
        ['FillerOrderNo', 'MRNCmpdId', 'start_time', 'slot_outcome']]

    result = pd.merge(d, r, left_on=['patient_id', 'start_time'], right_on=['MRNCmpdId', 'start_time'], how='outer',
                      suffixes=('_dispo', '_rdsc'))

    # create one patient id column that combines the dispo patient_id and slot_df patient_id
    result['id_or'] = np.where(~result['patient_id'].isna(), result['patient_id'], result['MRNCmpdId'])
    result['slot_outcome_dispo'].fillna('not present', inplace=True)
    result['slot_outcome_rdsc'].fillna('missing', inplace=True)

    error_pivot = pd.pivot_table(result, index='slot_outcome_dispo', columns=['slot_outcome_rdsc'], values='id_or',
                                 aggfunc='count')

    dispo_cols = columns.copy()
    rdsc_cols = columns.copy()
    if 'not present' in error_pivot.index:
        dispo_cols.extend(['not present'])
    if 'missing' in error_pivot.columns:
        rdsc_cols.extend(['missing'])
    return error_pivot.reindex(dispo_cols)[rdsc_cols]


def generate_data_firstexperiment_plot(dispo_data: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Iterates over unique dates in dispo_data (software data) and slot_df (extract)
    For each UNIQUE date, counts how many ["show","soft no-show","hard no-show"] there are.

    Args:
        dispo_data: Dataframe with appointment data
        slot_df: Dataframe with appointment data from extract

    Returns: dataframe that contains, date, year, num_shows, num_rescheduled, num_canceled , 'extract/experiment'

    '''

    df = pd.DataFrame(columns=['date', 'year', 'dispo_show', 'dispo_rescheduled', 'dispo_canceled',
                               'extract_show', 'extract_rescheduled', 'extract_canceled'])
    for date_elem in dispo_data.date.dt.date.unique():
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # Identify how many 'shows' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year, 'show')
        num_dispo_show = len(dispo_patids)
        num_extract_show = len(slot_df_patids)
        # Identify how many 'soft no-show' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   'rescheduled')
        num_dispo_rescheduled = len(dispo_patids)
        num_extract_rescheduled = len(slot_df_patids)
        # Identify how many 'hard no-show' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   'canceled')
        num_dispo_canceled = len(dispo_patids)
        num_extract_canceled = len(slot_df_patids)

        df = df.append({'date': date_elem,
                        'year': date_elem.year,
                        'dispo_show': num_dispo_show,
                        'dispo_rescheduled': num_dispo_rescheduled,
                        'dispo_canceled': num_dispo_canceled,
                        'extract_show': num_extract_show,
                        'extract_rescheduled': num_extract_rescheduled,
                        'extract_canceled': num_extract_canceled
                        }, ignore_index=True)

    return df


def calculate_ratios_experiment(df_experiment: pd.DataFrame, slot_outcome: str) -> pd.DataFrame:
    """
    Calculates ratios between number of show, rescheduled and canceled for plot

    Args:
        df_experiment: dataframe that contains num_shows, num_softnoshows,
             num_hard_noshow per date and if it is from extract or dispo
        slot_outcome: string with value ['show', 'rescheduled', 'canceled'].


    Returns: dataframe with three columns. These are ['date','year','ratio']. The last column being the
        ratio calculated between dispo and extract. If you specify 'show' as the type, then the resulting
        ratio column will have the ratio of 'show' appointments. This ratio is calculated as:
        ratio = # of shows in the extract / # of shows in the dispo data

    """

    if slot_outcome == 'show':
        drop_col = ['dispo_rescheduled', 'dispo_canceled', 'extract_rescheduled', 'extract_canceled']
        drop_col2 = ['extract_show', 'dispo_show']
    elif slot_outcome == 'rescheduled':
        drop_col = ['dispo_show', 'dispo_canceled', 'extract_show', 'extract_canceled']
        drop_col2 = ['extract_rescheduled', 'dispo_rescheduled']
    elif slot_outcome == 'canceled':
        drop_col = ['dispo_rescheduled', 'dispo_show', 'extract_rescheduled', 'extract_show']
        drop_col2 = ['extract_canceled', 'dispo_canceled']

    df_ratios = df_experiment.copy()
    df_ratios = df_ratios.drop(drop_col, axis=1)
    df_ratios['ratio'] = df_ratios[drop_col2[0]] / (df_ratios[drop_col2[1]] + 1)
    df_ratios = df_ratios.drop(drop_col2, axis=1)

    return df_ratios


def format_dicom_times_df(df):
    df['AccessionNumber'] = df['AccessionNumber'].astype(int)

    time_cols = ['image_start', 'image_end']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
    return df


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


def split_df_to_train_validate_test(df_input: pd.DataFrame, train_percent=0.7, validate_percent=0.15):
    """
    Args:
         df_input: dataframe with all variables of interest for the model

    Returns: dataframe with variables split into train, validation and test sets
    """

    df_output = df_input.copy()

    seed = 0
    np.random.seed(seed)
    perm = np.random.permutation(df_output.index)
    df_len = len(df_output.index)
    train_end = int(train_percent * df_len)
    validate_end = int(validate_percent * df_len) + train_end
    train = df_output.iloc[perm[:train_end]]
    validate = df_output.iloc[perm[train_end:validate_end]]
    test = df_output.iloc[perm[validate_end:]]

    return train, validate, test


def jaccard_index(dispo_set: Set, extract_set: Set) -> float:
    """
    Calculates the Jaccard Index for two given sets

    Args:
        dispo_set: set of ids identified in the dispo dataset for a given type
        extract_set: set of ids identified in the extract dataset for a given type

    Returns: score between 0.0 and 1.0, which is the Jaccard score.
    """
    score = 1.0
    if dispo_set or extract_set:
        score = (float(len(dispo_set.intersection(extract_set))) / len(dispo_set.union(extract_set)))

    return score


def jaccard_for_outcome(dispo_df: pd.DataFrame, slot_df: pd.DataFrame, slot_outcome: str) -> float:
    """
    Calculate the Jaccard score for a slot_outcome represented in both dispo_df and slot_df

    Args:
        dispo_df: result of `build_dispo_df`
        slot_df: result of `build_slot_df`
        slot_outcome:result of `set_slot_outcome`

    Returns: Jaccard score
    """
    dispo_dates = dispo_df['date'].dt.date.unique()
    slot_only_dates_df = slot_df[slot_df['start_time'].dt.date.isin(dispo_dates)]

    dispo_outcome_ids = dispo_df[dispo_df['slot_outcome'] == slot_outcome]['patient_id'].astype(str).sort_values()
    rdsc_outcome_ids = slot_only_dates_df[slot_only_dates_df['slot_outcome'] == slot_outcome]['MRNCmpdId'].astype(
        str).sort_values()

    return jaccard_index(set(dispo_outcome_ids), set(rdsc_outcome_ids))


def print_validation_summary_metrics(dispo_df, slot_df):
    """
    Print total slot counts from slot_df and dispo_df, and their average Jaccard index per day. Metrics are printed for
     separately show, rescheduled, and canceled slots.
    Args:
        dispo_df: result of build_dispo_df()
        slot_df: result fo build_slot_df()

    Returns: Dataframe with metrics

    """
    validation_dates = list(dispo_df.date.dt.date.unique())
    slot_patids = {}

    for outcome in ['show', 'rescheduled', 'canceled']:
        slot_patids[outcome] = {}

        total_slot_df_patids = []
        total_dispo_patids = []
        jaccard_indices = []
        for d in validation_dates:
            day, month, year = d.day, d.month, d.year
            dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_df, slot_df, day, month, year, outcome,
                                                                       verbose=False)
            total_slot_df_patids.extend(list(slot_df_patids))
            total_dispo_patids.extend(list(dispo_patids))
            jaccard_indices.append(jaccard_index(slot_df_patids, dispo_patids))

        slot_patids[outcome]['extract'] = total_slot_df_patids
        slot_patids[outcome]['dispo'] = total_dispo_patids
        slot_patids[outcome]['jaccard'] = jaccard_indices

    slot_cnts = {}
    for outcome in slot_patids:
        slot_cnts[outcome] = {}
        for source in ['extract', 'dispo']:
            slot_cnts[outcome][source] = len(slot_patids[outcome][source])
            slot_cnts[outcome]['jaccard'] = sum(slot_patids[outcome]['jaccard']) / len(slot_patids[outcome]['jaccard'])

    return pd.DataFrame(slot_cnts).T[['dispo', 'extract', 'jaccard']].style.format(
        {'dispo': '{:.0f}', 'extract': '{:.0f}', 'jaccard': '{:.2f}'})


def normalize_dataframe(df_features: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    '''
    Normalize columns in cols list

    Args:
        df_features: dataframe with features used in model
        cols: list of features on which min_max normalization is executed

    Returns:
        Normalized dataframes
    '''

    result = df_features.copy()
    for feature_name in cols:
        max_value = df_features[feature_name].max()
        min_value = df_features[feature_name].min()
        result[feature_name] = (df_features[feature_name] - min_value) / (max_value - min_value)
    return result
