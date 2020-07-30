"""
All processing functions for the data transformation pipeline.

### Major Data Processing Steps ###

build_status_df():
 - reads raw file from filesystem and adds custom columns.
 - This data is in the format one-row-per-appt-status-change

build_slot_df():
 - returns data in the form one-row-per-appointment-slot (now show or completed appointment)

"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Set


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
    df['patient_class_adj'] = df['PatientClass'].apply(adjust_patient_class)
    df['NoShow'] = df.apply(find_no_shows, axis=1)
    df['NoShow_severity'] = df.apply(set_no_show_severity, axis=1)
    df['slot_type'] = df.apply(set_slot_type, axis=1)
    df['slot_outcome'] = df.apply(set_no_show_outcome, axis=1)
    df['slot_type_detailed'] = df.apply(set_slot_type_detailed, axis=1)
    return df


def build_slot_df(input_status_df: pd.DataFrame, agg_dict: Dict[str, str] = None, include_id_cols: bool = True
                  ) -> pd.DataFrame:
    """
    Convert status_df into slot_df. Identify "show" and "no show" appointment slots from status_df,
    and synthesize into a single dataframe of all appointments that occurred or were supposed to occur (but no-show'ed).

    Args:
        input_status_df: row-per-status-change dataframe.
        agg_dict: aggregation dict to pass to pd.DataFrame.agg() that specifies  columns to include about the slots.
            If no agg_dict is passed, the default will be used.
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
         - MRNCmpdId (if available): int, patient id
    """

    default_agg_dict = {
        'start_time': 'min',
        'end_time': 'min',
        'NoShow': 'min',
        'slot_outcome': 'min',
        'slot_type': 'min',
        'slot_type_detailed': 'min',
        'EnteringOrganisationDeviceID': 'min',
        'UniversalServiceName': 'min',
    }
    if agg_dict is None:
        agg_dict = default_agg_dict

        if include_id_cols and 'MRNCmpdId' in input_status_df.columns:
            agg_dict['MRNCmpdId'] = 'min'

    status_df = input_status_df.copy()
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    status_df['start_time'] = status_df.apply(identify_start_times, axis=1)
    status_df['end_time'] = status_df.apply(identify_end_times, axis=1)
    status_df['end_time'] = status_df.groupby('FillerOrderNo')['end_time'].fillna(method='bfill')

    # there should be one show appt per FillerOrderNo
    show_slot_type_events = status_df[status_df['slot_type'].isin(['show', 'inpatient'])].copy()
    show_slot_df = show_slot_type_events.groupby(['FillerOrderNo']).agg(agg_dict).reset_index()

    # there may be multiple no-show appts per FillerOrderNo
    no_show_slot_type_events = status_df[status_df['NoShow']].copy()
    no_show_slot_df = no_show_slot_type_events.groupby(['FillerOrderNo', 'was_sched_for_date']).agg(
        agg_dict).reset_index()
    no_show_slot_df.drop('was_sched_for_date', axis=1, inplace=True)

    slot_df = pd.concat([show_slot_df, no_show_slot_df], sort=False)

    if not include_id_cols:
        slot_df.drop('FillerOrderNo', axis=1, inplace=True)

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
    elif row['slot_type'] == 'show':
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


def build_dispo_df(dispo_examples: List[Dict]) -> pd.DataFrame:
    """
    Convert dispo file to dataframe and converts date column to pd.datetime format.

    Args:
        dispo_examples: List of dictionaries containing appointment information

    Returns: Dataframe with datetime type conversions

    """
    dispo_df = pd.DataFrame(dispo_examples)
    dispo_df['patient_id'] = dispo_df['patient_id'].astype(int)
    dispo_df['start_time'] = pd.to_datetime(dispo_df['date'] + ' ' + dispo_df['start_time'], dayfirst=True)
    dispo_df['date'] = pd.to_datetime(dispo_df['date'], dayfirst=True)
    dispo_df['slot_outcome'] = np.where(dispo_df['type'] == 'show', 'show', dispo_df['slot_outcome'])

    return dispo_df


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


def generate_data_firstexperiment_plot(dispo_data: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Iterates over unique dates in dispo_data (software data) and slot_df (extract)
    For each UNIQUE date, counts how many ["show","soft no-show","hard no-show"] there are.

    Args:
        dispo_data: Dataframe with appointment data
        slot_df: Dataframe with appointment data from extract

    Returns: dataframe that contains, date, year, num_shows, num_softnoshows,
             num_hard_noshow , 'extract/experiment'

    '''

    # SNS stands for 'soft no-show' and HNS stands for 'hard no-show'
    df = pd.DataFrame(columns=['date', 'year', 'dispo_show', 'dispo_sns', 'dispo_hns',
                               'extract_show', 'extract_sns', 'extract_hns'])
    for date_elem in dispo_data.date.dt.date.unique():
        day, month, year = date_elem.day, date_elem.month, date_elem.year
        # Identify how many 'shows' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year, 'show')
        num_dispo_show = len(dispo_patids)
        num_extract_show = len(slot_df_patids)
        # Identify how many 'soft no-show' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   'soft no-show')
        num_dispo_sns = len(dispo_patids)
        num_extract_sns = len(slot_df_patids)
        # Identify how many 'hard no-show' in dispo_data and extract
        dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_data, slot_df, day, month, year,
                                                                   'hard no-show')
        num_dispo_hns = len(dispo_patids)
        num_extract_hns = len(slot_df_patids)

        df = df.append({'date': date_elem, 'year': date_elem.year, 'dispo_show': num_dispo_show,
                        'dispo_sns': num_dispo_sns, 'dispo_hns': num_dispo_hns, 'extract_show': num_extract_show,
                        'extract_sns': num_extract_sns, 'extract_hns': num_extract_hns}, ignore_index=True)

    return df


def calculate_ratios_experiment(df_experiment: pd.DataFrame, slot_outcome: str) -> pd.DataFrame:
    """
    Calculates ratios between number of show, soft no-show and hard no-show for plot

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
        drop_col = ['dispo_resched', 'dispo_cancel', 'extract_resched', 'extract_cancel']
        drop_col2 = ['extract_show', 'dispo_show']
    elif slot_outcome == 'rescheduled':
        drop_col = ['dispo_show', 'dispo_cancel', 'extract_show', 'extract_cancel']
        drop_col2 = ['extract_resched', 'dispo_resched']
    elif slot_outcome == 'canceled':
        drop_col = ['dispo_resched', 'dispo_show', 'extract_resched', 'extract_show']
        drop_col2 = ['extract_cancel', 'dispo_cancel']

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
