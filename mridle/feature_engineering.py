import pandas as pd
import numpy as np


def find_end_times(row):
    if row['now_status'] == 'examined':
        return row['date']
    else:
        return None


def feature_scheduled_for_hour(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['sched_for_hour'] = status_df['was_sched_for_date'].dt.hour
    return status_df


def calc_days_sched_in_advance(row):
    if row['was_sched_for'] != row['now_sched_for']:
        return row['now_sched_for']
    else:
        return None


def feature_days_scheduled_in_advance(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['days_sched_in_advance'] = status_df.apply(calc_days_sched_in_advance, axis=1)
    status_df['days_sched_in_advance'] = status_df.groupby('FillerOrderNo')['days_sched_in_advance'].fillna(
        method='ffill')
    return status_df


def feature_day_of_week(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['day_of_week'] = status_df['was_sched_for_date'].dt.dayofweek
    return status_df


def feature_modality(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['modality'] = status_df['UniversalServiceName']
    return status_df


def feature_marital(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapping from https://de.wikipedia.org/wiki/Familienstand
    Args:
        status_df:

    Returns:

    """
    zivilstand_abbreviation_mapping = {
        'VRH': 'married',
        'LED': 'single',
        'GES': 'divorced',
        'UNB': 'not known',
        'VRW': 'widowed',
        'GTR': 'unable to translate',
        'PAR': 'partnership',
        # 'EA': 'marriage canceled',
        # 'LP': 'in registered civil partnership',
        # 'LV': 'life partnership dissolved by death',
        # 'LA': 'forcible partnership',
        # 'LE': 'civil partnership dissolved by declaration of death',
        np.NaN: 'blank',
    }
    status_df['marital'] = status_df['Zivilstand'].apply(lambda x: zivilstand_abbreviation_mapping[x])
    return status_df


def feature_distance_to_usz(status_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: get zip distance
    status_df['distance_to_usz'] = status_df['WohnadrPLZ']
    return status_df


def feature_historic_no_show_count(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['historic_no_show_cnt'] = status_df.groupby('MRNCmpdId')['NoShow'].cumsum()
    return status_df


def build_harvey_et_al_features_set(status_df: pd.DataFrame, drop_id_col=True) -> pd.DataFrame:
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    # status_df['end_time'] = status_df.apply(find_end_times, axis=1)
    # status_df['end_time'] = status_df.groupby('FillerOrderNo')['end_time'].fillna(method='bfill')

    status_df = feature_scheduled_for_hour(status_df)
    status_df = feature_days_scheduled_in_advance(status_df)
    status_df = feature_day_of_week(status_df)
    status_df = feature_modality(status_df)
    status_df = feature_marital(status_df)
    status_df = feature_distance_to_usz(status_df)
    status_df = feature_historic_no_show_count(status_df)

    # re-shape into slot_df
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])
    show_slot_status_events = status_df[(status_df['PatientClass'] == 'ambulent') & (status_df['OrderStatus'] == 'u') &
                                        (status_df['now_status'] == 'started')].copy()
    no_show_slot_status_events = status_df[status_df['NoShow']].copy()

    agg_dict = {
        'NoShow': 'min',
        'sched_for_hour': 'first',
        'days_sched_in_advance': 'first',
        'modality': 'last',
        'day_of_week': 'last',
        'marital': 'last',
        'distance_to_usz': 'last',
        'historic_no_show_cnt': 'last',
    }

    # there should be one show appt per FillerOrderNo
    show_slot_df = show_slot_status_events.groupby(['FillerOrderNo']).agg(agg_dict).reset_index()

    # there may be multiple no-show appts per FillerOrderNo
    no_show_slot_df = no_show_slot_status_events.groupby(['FillerOrderNo', 'start_time']).agg(
        agg_dict).reset_index()
    no_show_slot_df.drop('start_time', inplace=True)

    new_slot_df = pd.concat([show_slot_df, no_show_slot_df])

    if drop_id_col:
        new_slot_df.drop('FillerOrderNo', inplace=True)

    return new_slot_df
