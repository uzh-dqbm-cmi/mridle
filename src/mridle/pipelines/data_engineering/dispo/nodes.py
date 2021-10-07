from typing import Dict, List, Union

import numpy as np
import pandas as pd
from mridle.utilities import data_processing


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


# def calc_jaccard_score_table(data: ValDataTuple) -> pd.DataFrame:
#     """
#     Calculate the Jaccard scores for show, rescheduled, and canceled appointments in the Development and Evaluation
#      sets and generate a table.
#
#     Args:
#         data: Dispo and RIS data object.
#
#     Returns: Dataframe with rows for show, rescheduled, and canceled appointments, and columns for development and
#      evaluation experiments, and values of the Jaccard scores.
#
#     """
#     dispo_data, ris_data = data
#     jaccard_results = {
#         'development': {
#             'show': 0,
#             'canceled': 0,
#             'rescheduled': 0,
#         },
#         'evaluation': {
#             'show': 0,
#             'canceled': 0,
#             'rescheduled': 0,
#         }
#     }
#
#     for exp in experiments:
#         dispo_slot_df = dispo_data['slot_df'][exp]
#         ris_slot_df = ris_data['slot_df'][exp]
#         for appt_type in ['show', 'canceled', 'rescheduled']:
#             jaccard_results[exp][appt_type] = mridle.data_management.jaccard_for_outcome(dispo_slot_df, ris_slot_df,
#                                                                                          appt_type)
#
#     jaccard_results_df = pd.DataFrame(jaccard_results)
#     return jaccard_results_df


def color_red(val):
    if val > 0:
        return 'color: red'
    else:
        return 'color: black'


def color_orange(val):
    if val > 0:
        return 'color: orange'
    else:
        return 'color: black'


def color_green(val):
    if val > 0:
        return 'color: green'
    else:
        return 'color: black'


def calc_exp_confusion_matrix(val_dispo_slot_df: pd.DataFrame, val_ris_slot_df: pd.DataFrame
                              ):
    """
    Create a styled dataframe of the confusion matrix for either the development or evaluation experiment.

    Args:
        val_dispo_slot_df: slot_df for the dispo data to validate.
        val_ris_slot_df: slot_df for the ris data to validate the dispo data against.

    Returns: Pandas dataframe styled with color coding.

    """
    # -> pd.io.formats.style.Styler:

    c = validation_exp_confusion_matrix(val_dispo_slot_df, val_ris_slot_df, ['show', 'rescheduled', 'canceled']
                                        ).fillna(0).astype(int)
    c.loc['not present', 'missing'] = 0
    c = c.style.applymap(color_red, subset=pd.IndexSlice['rescheduled', ['show']])
    c = c.applymap(color_red, subset=pd.IndexSlice['show', ['rescheduled']])
    c = c.applymap(color_red, subset=pd.IndexSlice['show', ['canceled']])
    c = c.applymap(color_orange, subset=pd.IndexSlice['not present', ])
    c = c.applymap(color_orange, subset=pd.IndexSlice[:, ['missing']])
    c = c.applymap(color_green, subset=pd.IndexSlice['show', ['show']])
    c = c.applymap(color_green, subset=pd.IndexSlice['rescheduled', ['rescheduled']])
    c = c.applymap(color_green, subset=pd.IndexSlice['canceled', ['canceled']])
    return c


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
    r['MRNCmpdId'] = r['MRNCmpdId'].astype(str)

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
