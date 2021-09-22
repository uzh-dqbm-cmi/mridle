import pandas as pd
import numpy as np
from mridle.pipelines.data_engineering.ris.nodes import build_slot_df
import pgeocode
import datetime as dt


def build_feature_set(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a feature set that replicates the Harvey et al model as best we can.
    So far includes:
        - sched_days_advanced: Number of days the appt was scheduled in advance
        - day_of_week: The day of the week of the appt (1=Monday)
        - modality: The UniversalServiceName of the appt
        - marital: Zivilstand of the patient
        - distance_to_usz: distance from the patient's home address to the hospital, approximated from Post Codes
        - no_show_before: The number of no shows the patient has had up to the date of the appt
    Args:
        status_df:

    Returns:

    """
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    status_df = feature_month(status_df)
    status_df = feature_hour_sched(status_df)
    status_df = feature_day_of_week(status_df)
    status_df = feature_days_scheduled_in_advance(status_df)
    status_df = feature_modality(status_df)
    status_df = feature_insurance_class(status_df)
    status_df = feature_sex(status_df)
    status_df = feature_age(status_df)
    status_df = feature_marital(status_df)
    status_df = feature_post_code(status_df)
    status_df = feature_distance_to_usz(status_df)
    status_df = feature_no_show_before(status_df)

    agg_dict = {
        'NoShow': 'min',
        'hour_sched': 'first',
        'sched_days_advanced': 'first',
        'modality': 'last',
        'insurance_class': 'last',
        'day_of_week': 'last',
        'day_of_week_str': 'last',
        'month': 'last',
        'sex': 'last',
        'age': 'last',
        'marital': 'last',
        'post_code': 'last',
        'distance_to_usz': 'last',
        'no_show_before': 'last',
        'slot_outcome': 'last',
        'date': 'last'
    }

    slot_df = build_slot_df(status_df, agg_dict, include_id_cols=True)

    return slot_df


# Feature engineering functions
def identify_end_times(row: pd.DataFrame) -> dt.datetime:
    """
    Identify end times of show appts. Could be used like this:
      status_df['end_time'] = status_df.apply(identify_end_times, axis=1)
      status_df['end_time'] = status_df.groupby('FillerOrderNo')['end_time'].fillna(method='bfill')

    Args:
        row: row from a database, as generated by using df.apply(axis=1).

    Returns: appt end datetime, or None if the row is not an appt ending event.

    """
    if row['now_status'] == 'examined':
        return row['date']
    else:
        return None


def feature_month(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the day_of_week feature to the dataframe.

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'month' containing integers 1-12.

    """
    status_df['month'] = status_df['was_sched_for_date'].dt.month
    return status_df


def feature_hour_sched(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the hour_sched feature to the dataframe using was_sched_for_date.

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'hour_sched'.
    """
    status_df['hour_sched'] = status_df['was_sched_for_date'].dt.hour
    return status_df


def feature_day_of_week(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the day_of_week feature to the dataframe.

    Args:
        status_df: A row-per-status-change dataframe.

    Returns:
        A row-per-status-change dataframe with additional columns 'day_of_week' (containing integers 0-6)
        and `day_of_week_str` containing strings in the format 'Monday', 'Tuesday', ...

    """
    status_df['day_of_week'] = status_df['was_sched_for_date'].dt.dayofweek
    status_df['day_of_week_str'] = status_df['was_sched_for_date'].dt.strftime('%A')

    return status_df


def identify_sched_events(row: pd.DataFrame) -> dt.datetime:
    """
    Identify scheduling events, for use in feature_days_scheduled_in_advance.

    Args:
        row: row: A row from a database, as generated by using df.apply(axis=1).

    Returns: scheduling datetime, or None if the row is not a scheduling event.

    """
    if row['was_sched_for'] != row['now_sched_for']:
        return row['now_sched_for']
    else:
        return None


def feature_days_scheduled_in_advance(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the sched_days_advanced feature to the dataframe.
    Works by:
        1. Identify status changes that represent scheduling events
        2. Shift scheduling events forward 1, so that each row has the previous scheduling event.
            For example, on a No-Show status change row, Step 1 will stamp the scheduling event that occurs as a result
             of a no-show going from scheduled status -> scheduled status. To calculate the scheduled date of the
              no-show appt slot, we need the previous scheduling event.
        3. Fill forward so the scheduling event dates so that 'show' and 'no-show' appt status rows contain the date of
         the most recent (but previous) scheduling event.

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'sched_days_advanced'.
    """
    status_df['sched_days_advanced'] = status_df.apply(identify_sched_events, axis=1)
    status_df['sched_days_advanced'] = status_df.groupby('FillerOrderNo')['sched_days_advanced'].shift(1).fillna(
        method='ffill')
    return status_df


def feature_modality(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the modality feature to the dataframe.

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'marital'.

    """
    status_df['modality'] = status_df['UniversalServiceName']
    return status_df


def feature_insurance_class(status_df: pd.DataFrame) -> pd.DataFrame:
    insurance_class_map = {
        'A': 'general',
        'P': 'private',
        'HP': 'half private',
    }
    status_df['insurance_class'] = status_df['Klasse'].apply(lambda x: insurance_class_map.get(x, 'unknown'))
    return status_df


def feature_sex(status_df: pd.DataFrame) -> pd.DataFrame:
    gender_map = {
        'weiblich': 'female',
        'männlich': 'male',
        'unbekannt': 'unknown',
    }
    status_df['sex'] = status_df['Sex'].apply(lambda x: gender_map.get(x, 'unknown'))
    return status_df


def feature_age(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['age'] = pd.to_datetime(status_df['date']).dt.year - pd.to_datetime(status_df['DateOfBirth']).dt.year
    return status_df


def feature_marital(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Label teh Zivilstand of the patient in English.
    Mapping from https://de.wikipedia.org/wiki/Familienstand

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'marital'.

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
        'XXX': 'undefined',
    }

    status_df['marital'] = status_df['Zivilstand'].map(zivilstand_abbreviation_mapping)
    return status_df


def feature_post_code(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['post_code'] = status_df['Zip']
    return status_df


def feature_distance_to_usz(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance between the patient's home post code and the post code of the hospital.

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'distance_to_usz'.
    """
    dist = pgeocode.GeoDistance('ch')
    usz_post_code = '8091'
    status_df['post_code'] = status_df['post_code'].astype(str).replace("\\.0", '', regex=True)

    unique_zips = pd.DataFrame(status_df['post_code'].unique(), columns=['post_code'])
    unique_zips['distance_to_usz'] = unique_zips['post_code'].apply(lambda x: dist.query_postal_code(x, usz_post_code))
    status_df = pd.merge(status_df, unique_zips, on='post_code', how='left')
    return status_df


def feature_no_show_before(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    The number of no-shows the patient has had up to and _not including_ this one.
    Historic no show counts are limited to the bounds of the dataset- it does not include no-shows not included in the
     present dataset.
    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional column 'no_show_before'.

    """
    status_df['no_show_before'] = status_df.groupby('MRNCmpdId')['NoShow'].cumsum()
    # cumsum will include the current no show, so subtract 1, except don't go negative
    status_df['no_show_before'] = np.where(status_df['no_show_before'] > 0, status_df['no_show_before'] - 1, 0)
    return status_df
