"""
All processing functions for the data transformation pipeline.

### Major Data Processing Steps ###

build_status_df():
 - reads raw file from filesystem and adds custom columns.
 - This data is in the format one-row-per-appt-status-change

build_slot_df():
 - returns data in the form one-row-per-appointment-slot (a slot is a no show or completed appointment)

"""

from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd
from mridle.utilities import data_processing


def aggregate_terminplanner(terminplanner_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in raw terminplanner data, where each row represents one possible appointment slot
    (e.g.   Terminbuch  Wochentag   TERMINRASTER_NAME      gültig von  gültig bis      Termin      Dauer in Min.
            MR1         DO          MR1 IDR (Donnerstag)   05.12.2018  20.02.2019      07:00       35
            MR1         DO          MR1 IDR (Donnerstag)   05.12.2018  20.02.2019      07:35       35

    )
    and returns an aggregated representation of this data, with each row representing one day of the week, and the
    start and end time of the window for appointments.

    Args:
        terminplanner_df: Raw terminplanner data, provided by Beat Hümbelin

    Returns:
        tp_agg, a pd.DataFrame with a row for each day of the week/machine combination, with information on the
        starting and finishing time for the MR machine, along with a date range for which these times are applicable
        for. A column containing the total number of minutes in the day is included as well.
    """
    tp_df = terminplanner_df.copy()
    tp_df['Termin'] = pd.to_datetime(tp_df['Termin'], format='%H:%M')
    tp_df['Terminbuch'] = tp_df['Terminbuch'].replace({'MR1': 1, 'MR2': 2})
    tp_df['Wochentag'] = tp_df['Wochentag'].replace({'MO': 'Monday',
                                                     'DI': 'Tuesday',
                                                     'MI': 'Wednesday',
                                                     'DO': 'Thursday',
                                                     'FR': 'Friday'
                                                     })
    tp_df['Dauer in dt'] = pd.to_timedelta(tp_df['Dauer in Min.'], unit='m')
    tp_df['terminende'] = tp_df['Termin'] + tp_df['Dauer in dt']
    tp_df['Termin'] = tp_df['Termin'].dt.time
    tp_df['terminende'] = tp_df['terminende'].dt.time
    tp_agg = tp_df.groupby(['Terminbuch', 'Wochentag', 'TERMINRASTER_NAME', 'gültig von', 'gültig bis']).agg({
        'Termin': 'min',
        'terminende': 'max',
        'Dauer in Min.': 'sum'
    }).reset_index()

    tp_agg.rename(columns={'Wochentag': 'day_of_week',
                           'Terminbuch': 'image_device_id',
                           'Termin': 'day_start_tp',
                           'terminende': 'day_end_tp',
                           'Dauer in Min.': 'day_length_tp',
                           'gültig von': 'applicable_from',
                           'gültig bis': 'applicable_to'}, inplace=True)
    return tp_agg


# ========================================================================================
# === HELPER FUNCTIONS ===================================================================
# ========================================================================================

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
    deduped_dispo_slot_df = data_processing.filter_duplicate_patient_time_slots(dispo_slot_df)
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
    deduped_dispo_slot_df = data_processing.filter_duplicate_patient_time_slots(dispo_slot_df)
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
    # advance of the appt, and then pick the last occurrence by applying a cumcount to the descending ordered list and
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
                       )
    one_day = one_day.sort_values(['start_time'])

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
            if first_status_after == 'bef':
                return 'show'
            else:
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
    """
    Iterates over unique dates in dispo_data (software data) and slot_df (extract)
    For each UNIQUE date, counts how many ["show","soft no-show","hard no-show"] there are.

    Args:
        dispo_data: Dataframe with appointment data
        slot_df: Dataframe with appointment data from extract

    Returns: dataframe that contains, date, year, num_shows, num_rescheduled, num_canceled , 'extract/experiment'

    """

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
