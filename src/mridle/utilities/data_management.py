"""
All processing functions for the data transformation pipeline.

### Major Data Processing Steps ###

build_status_df():
 - reads raw file from filesystem and adds custom columns.
 - This data is in the format one-row-per-appt-status-change

build_slot_df():
 - returns data in the form one-row-per-appointment-slot (a slot is a no show or completed appointment)

"""

from typing import List, Set

import numpy as np
import pandas as pd


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


# def print_validation_summary_metrics(dispo_df, slot_df):
#     """
#     Print total slot counts from slot_df and dispo_df, and their average Jaccard index per day. Metrics are printed
#      for separately show, rescheduled, and canceled slots.
#     Args:
#         dispo_df: result of build_dispo_df()
#         slot_df: result fo build_slot_df()
#
#     Returns: Dataframe with metrics
#
#     """
#     validation_dates = list(dispo_df.date.dt.date.unique())
#     slot_patids = {}
#
#     for outcome in ['show', 'rescheduled', 'canceled']:
#         slot_patids[outcome] = {}
#
#         total_slot_df_patids = []
#         total_dispo_patids = []
#         jaccard_indices = []
#         for d in validation_dates:
#             day, month, year = d.day, d.month, d.year
#             dispo_patids, slot_df_patids = validate_against_dispo_data(dispo_df, slot_df, day, month, year, outcome,
#                                                                        verbose=False)
#             total_slot_df_patids.extend(list(slot_df_patids))
#             total_dispo_patids.extend(list(dispo_patids))
#             jaccard_indices.append(jaccard_index(slot_df_patids, dispo_patids))
#
#         slot_patids[outcome]['extract'] = total_slot_df_patids
#         slot_patids[outcome]['dispo'] = total_dispo_patids
#         slot_patids[outcome]['jaccard'] = jaccard_indices
#
#     slot_cnts = {}
#     for outcome in slot_patids:
#         slot_cnts[outcome] = {}
#         for source in ['extract', 'dispo']:
#             slot_cnts[outcome][source] = len(slot_patids[outcome][source])
#             slot_cnts[outcome]['jaccard'] = sum(slot_patids[outcome]['jaccard']) / \
#                                             len(slot_patids[outcome]['jaccard'])
#
#     return pd.DataFrame(slot_cnts).T[['dispo', 'extract', 'jaccard']].style.format(
#         {'dispo': '{:.0f}', 'extract': '{:.0f}', 'jaccard': '{:.2f}'})


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
