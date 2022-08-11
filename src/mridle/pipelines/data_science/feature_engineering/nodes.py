import pandas as pd
import numpy as np
from mridle.pipelines.data_engineering.ris.nodes import build_slot_df
import pgeocode
import datetime as dt
import re
from sklearn.model_selection import train_test_split
from typing import Dict, List


def build_model_data(status_df, valid_date_range, slot_df=None):
    """
    Build data for use in models by trying to replicate the conditions under which the model would be used in reality
    (i.e. no status changes 2 days before appt (since that's when the prediction would be done)). We then use the
    previously created slot_df (i.e. master appointment list) to filter the appts for only those that are relevant - the
    build_feature_set function with build_future_slots=True will create too many appointment slots, so we that's why we
    have to filter. We also need to use slot_df to get the outcome of the appointment, since build_future_slots=True,
    results in all appts appearing as NoShow=False. (replicating what would happen in reality...we would predict more
    than 2 days in advance, then wait and find out the outcome and join it onto our predictions)

    Args:
        status_df:
        slot_df:
        valid_date_range:

    Returns:

    """
    # valid_date_range = catalog.load('params:ris.valid_date_range')
    status_df_copy = status_df.copy()
    status_df_copy = status_df_copy[status_df_copy['now_sched_for'] > 2]

    model_data = build_feature_set(status_df_copy, valid_date_range=valid_date_range, build_future_slots=True)
    model_data = remove_na(model_data)
    if slot_df is not None:
        model_data.drop('NoShow', axis=1, inplace=True)
        # slot_df = catalog.load('slot_df')
        slot_df_copy = slot_df.copy()[['MRNCmpdId', 'FillerOrderNo', 'start_time', 'patient_class_adj', 'NoShow',
                                       'slot_outcome', 'slot_type', 'slot_type_detailed']]
        model_data = model_data.merge(slot_df_copy, how='inner')
    else:  # If slot_df not provided, then we are generating data for the future (e.g. Silent Live Test), therefore
        # NoShow should be false for all appointments (since they're future appts)

        # Remove appts where last status is 'canceled'
        last_status = status_df_copy.groupby(['FillerOrderNo']).apply(
            lambda x: x.sort_values('History_MessageDtTm', ascending=False).head(1)
        ).reset_index(drop=True)[['MRNCmpdId', 'FillerOrderNo', 'now_status', 'now_sched_for_busday']]

        last_status['remove'] = np.where(last_status['now_status'] == 'canceled' and
                                         last_status['now_sched_for_busday'] > 2, True, False)
        fon_to_remove = last_status.loc[last_status['remove'], 'FillerOrderNo']

        status_df_copy = status_df_copy[~status_df_copy['FillerOrderNo'].isin(fon_to_remove)]

        appt_time = status_df_copy.groupby(['FillerOrderNo']).apply(
            lambda x: x.sort_values('History_MessageDtTm', ascending=False).head(1)
        ).reset_index(drop=True)[['MRNCmpdId', 'FillerOrderNo', 'now_sched_for_date']]
        appt_time.columns = ['MRNCmpdId', 'FillerOrderNo', 'start_time']
        model_data = model_data.merge(appt_time, how='inner')
        model_data['NoShow'] = False
    return model_data


def build_feature_set(status_df: pd.DataFrame, valid_date_range: List[str], master_slot_df: pd.DataFrame = None,
                      build_future_slots: bool = True) -> pd.DataFrame:
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
        master_slot_df:
        build_future_slots:
        status_df: status_df
        valid_date_range: List of 2 strings defining the starting date of the valid slot data period (status_df contains
         status change data outside the valid slot date range- these should not be made into slots).
         build_future_slots: whether we are building slot_df for appointments in the future to build dataset for
            predictions (and therefore no show/no-show type events yet), or we are building it 'normally' (with past
            data and show/no-show events) to train models with.
    Returns:

    """
    status_df = status_df.sort_values(['FillerOrderNo', 'date'])

    status_df = status_df[status_df['patient_class_adj'] != 'inpatient']

    status_df = feature_modality(status_df)
    status_df = feature_insurance_class(status_df)
    status_df = feature_sex(status_df)
    status_df = feature_age(status_df)
    status_df = feature_marital(status_df)
    status_df = feature_post_code(status_df)
    status_df = feature_distance_to_usz(status_df)
    status_df = feature_occupation(status_df)
    status_df = feature_reason(status_df)

    agg_dict = {
        'NoShow': 'min',
        'modality': 'last',
        'occupation': 'last',
        'reason': 'last',
        'insurance_class': 'last',
        'sex': 'last',
        'age': 'last',
        'age_sq': 'last',
        'age_20_60': 'last',
        'marital': 'last',
        'post_code': 'last',
        'distance_to_usz': 'last',
        'distance_to_usz_sq': 'last',
        'close_to_usz': 'last',
        'start_time': 'last'
    }

    slot_df = build_slot_df(status_df, valid_date_range, agg_dict, build_future_slots=build_future_slots,
                            include_id_cols=True)

    slot_df = feature_days_scheduled_in_advance(status_df, slot_df)
    slot_df = feature_month(slot_df)
    slot_df = feature_hour_sched(slot_df)
    slot_df = feature_day_of_week(slot_df)
    slot_df = feature_no_show_before(slot_df)
    slot_df = feature_cyclical_hour(slot_df)
    slot_df = feature_cyclical_day_of_week(slot_df)
    slot_df = feature_cyclical_month(slot_df)
    slot_df = slot_df[slot_df['day_of_week_str'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
    slot_df = slot_df[slot_df['sched_days_advanced'] > 2]

    return slot_df


def remove_na(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Changes variables for model optimization modifying feature_df

    Args:
        dataframe: dataframe obtained from feature generation

    Returns: modified dataframe specific for this model
    """

    dataframe = dataframe.dropna(axis=0).reset_index(drop=True)

    return dataframe


def train_val_split(df: pd.DataFrame, params: Dict):
    test_data, validation_data = train_test_split(df, test_size=params['test_size'], random_state=94)
    return test_data, validation_data


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


def feature_month(slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the month feature to the dataframe.

    Args:
        slot_df: A dataframe containing appointment slots.

    Returns: A row-per-status-change dataframe with additional column 'month' containing integers 1-12.

    """
    slot_df['month'] = slot_df['start_time'].dt.month
    return slot_df


def feature_hour_sched(slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the hour_sched feature to the dataframe using was_sched_for_date.

    Args:
        slot_df: A dataframe containing appointment slots.

    Returns: A row-per-status-change dataframe with additional column 'hour_sched'.
    """
    slot_df['hour_sched'] = slot_df['start_time'].dt.hour
    return slot_df


def feature_day_of_week(slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the day_of_week feature to the dataframe.

    Args:
        slot_df: A dataframe containing appointment slots.

    Returns:
        A row-per-status-change dataframe with additional columns 'day_of_week' (containing integers 0-6)
        and `day_of_week_str` containing strings in the format 'Monday', 'Tuesday', ...

    """
    slot_df['day_of_week'] = slot_df['start_time'].dt.dayofweek
    slot_df['day_of_week_str'] = slot_df['start_time'].dt.strftime('%A')

    return slot_df


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


def feature_days_scheduled_in_advance(status_df: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the features 'sched_days_advanced' (int), 'sched_days_advanced_busday' (int), 'sched_days_advanced_sq' (int)
    and 'sched_2_days' (bool) to slot_df.

    Works by:
        Taking all rows from status_df where the date of the appointment changes. These are then grouped and the number
        of days in advance this date change was made is calculated. This is then joined onto the slot_df for each
        appointment.

    Args:
        status_df: A row-per-status-change dataframe.
        slot_df: A dataframe containing appointment slots.

    Returns: A row-per-status-change dataframe with additional columns 'sched_days_advanced', 'sched_days_advanced_sq'
    and 'sched_2_days'.
    """

    status_df['date_scheduled_change'] = (status_df['was_sched_for_date'] != status_df['now_sched_for_date'])
    date_changed = status_df.loc[status_df['date_scheduled_change'],
                                 ['FillerOrderNo', 'now_sched_for_date', 'now_sched_for', 'now_sched_for_busday']]
    days_advanced_schedule = date_changed.groupby(['FillerOrderNo', 'now_sched_for_date']).agg({
        'now_sched_for': 'first',
        'now_sched_for_busday': 'first'
    }).reset_index()
    days_advanced_schedule.columns = ['FillerOrderNo', 'now_sched_for_date', 'sched_days_advanced',
                                      'sched_days_advanced_busday']
    slot_df = slot_df.merge(days_advanced_schedule, left_on=['FillerOrderNo', 'start_time'],
                            right_on=['FillerOrderNo', 'now_sched_for_date'])
    slot_df.drop('now_sched_for_date', axis=1, inplace=True)
    slot_df['sched_days_advanced_sq'] = slot_df['sched_days_advanced'] ** 2
    slot_df['sched_2_days'] = slot_df['sched_days_advanced'] <= 2

    return slot_df


def feature_insurance_class(status_df: pd.DataFrame) -> pd.DataFrame:
    insurance_class_map = {
        'A': 'general',
        'P': 'private',
        'HP': 'half private',
    }
    status_df['insurance_class'] = status_df['Klasse'].apply(lambda x: insurance_class_map.get(x, 'unknown'))
    return status_df


def feature_sex(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['sex'] = np.where(status_df['Sex'] == 'weiblich', 1, 0)
    return status_df


def feature_age(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['age'] = pd.to_datetime(status_df['date']).dt.year - pd.to_datetime(status_df['DateOfBirth']).dt.year
    status_df['age_sq'] = status_df['age'] ** 2
    status_df['age_20_60'] = (status_df['age'] > 20) & (status_df['age'] < 60)

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
        'XXX': 'not known',
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

    status_df['marital'] = status_df['Zivilstand'].map(zivilstand_abbreviation_mapping)
    return status_df


def feature_post_code(status_df: pd.DataFrame) -> pd.DataFrame:
    status_df['post_code'] = status_df['Zip']
    return status_df


def feature_distance_to_usz(status_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance between the patient's home post code and the post code of the hospital. After calculating this,
    add a feature which is the distance_squared (used in harvey models) and then a boolean indicating whether the
    patient is 'close' to the hospital

    Args:
        status_df: A row-per-status-change dataframe.

    Returns: A row-per-status-change dataframe with additional columns 'distance_to_usz', 'distance_to_usz_sq', and
    'close_to_usz'.
    """
    dist = pgeocode.GeoDistance('ch')
    usz_post_code = '8091'
    status_df['post_code'] = status_df['post_code'].astype(str).replace("\\.0", '', regex=True)

    unique_zips = pd.DataFrame(status_df['post_code'].unique(), columns=['post_code'])
    unique_zips['distance_to_usz'] = unique_zips['post_code'].apply(lambda x: dist.query_postal_code(x, usz_post_code))
    status_df = pd.merge(status_df, unique_zips, on='post_code', how='left')
    status_df['distance_to_usz_sq'] = status_df['distance_to_usz'] ** 2
    status_df['close_to_usz'] = status_df['distance_to_usz'] < 16

    return status_df


def feature_no_show_before(slot_df: pd.DataFrame) -> pd.DataFrame:
    """
    The number of no-shows the patient has had up to and _not including_ this one.
    Historic no show counts are limited to the bounds of the dataset- it does not include no-shows not included in the
     present dataset.
    Args:
        slot_df: A row-per-appointment dataframe.

    Returns: A row-per-appointment dataframe with additional columns 'no_show_before', 'no_show_before_sq'.

    """
    slot_df_ordered = slot_df.sort_values('start_time')
    slot_df_ordered['no_show_before'] = slot_df_ordered.groupby('MRNCmpdId')['NoShow'].cumsum()
    # cumsum will include the current no show, so subtract 1, except don't go negative
    slot_df_ordered['no_show_before'] = np.where(slot_df_ordered['NoShow'], slot_df_ordered['no_show_before'] - 1,
                                                 slot_df_ordered['no_show_before'])

    slot_df_ordered['no_show_before_sq'] = slot_df_ordered['no_show_before'] ** 2

    return slot_df_ordered


def feature_modality(slot_df: pd.DataFrame, group_categories_less_than: int = None) -> pd.DataFrame:
    """
    Renames UniversalServiceName to modality, and maps this column to more general groups, defined by us.

    Args:
        slot_df: A row-per-appointment dataframe.
        group_categories_less_than: If provided, we remap all the remapped categories with fewer than the user-chosen
            number of occurrences/rows to 'other'

    Returns:
        dataframe with modality column added, and mapping applied to this column.
    """

    df_remap = slot_df.copy()
    df_remap['modality'] = ""

    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="becken"), 'modality'] = 'back'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="leber"), 'modality'] = 'liver'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str='niere'), 'modality'] = 'kidney'
    df_remap.loc[df_remap['UniversalServiceName'].apply(
        regex_search, search_str='hand|finger|ellbogen|vorderarm|oberarm|obere extremität'), 'modality'] = 'arm'
    df_remap.loc[df_remap['UniversalServiceName'].apply(
        regex_search, search_str="abdomen|thorax|hüfte|MR TOS"), 'modality'] = 'midsection'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="schenkel"), 'modality'] = 'leg'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="ganzkörper|ganzkvrper"),
                 'modality'] = 'full_body'
    df_remap.loc[df_remap['UniversalServiceName'].apply(
        regex_search, search_str="schädel|schadel|gehirn|felsenbein"), 'modality'] = 'head'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="herz"), 'modality'] = 'heart'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search,
                                                        search_str="Pankreas|Dünndarm|Milz|MRCP"), 'modality'] = 'organ'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="Intervention"),
                 'modality'] = 'intervention'
    df_remap.loc[df_remap['UniversalServiceName'].apply(
        regex_search, search_str="Neurographie|Magnetresonanztomographie"), 'modality'] = 'general'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="Angio"), 'modality'] = 'angiography'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="Arthrographie"), 'modality'] = 'joint'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search,
                                                        search_str="venograp|Phlebographie"), 'modality'] = 'veins'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="Mamma"), 'modality'] = 'mammography'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="Prostata"), 'modality'] = 'prostate'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="Hals"), 'modality'] = 'throat'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search,
                                                        search_str="Defäkographie"), 'modality'] = 'defecography'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="LWS|BWS|HWS"), 'modality'] = 'spine'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="schulter"), 'modality'] = 'shoulder'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="knie"), 'modality'] = 'knee'
    df_remap.loc[df_remap['UniversalServiceName'].apply(regex_search, search_str="fuss"), 'modality'] = 'foot'
    df_remap.loc[df_remap['modality'] == "", 'modality'] = 'other'

    return df_remap


def feature_time_of_day(slot_df):
    """
    Categorises the 'hour_sched' column into buckets.

    Args:
        slot_df: A row-per-appointment dataframe.

    Returns: A row-per-appointment dataframe with additional column 'time_of_day'.

    """

    df_copy = slot_df.copy()
    df_copy['time_of_day'] = pd.cut(df_copy['hour_sched'], bins=[-1, 9, 12, 14, 17, 100],
                                    labels=['early_morning', 'late_morning', 'lunchtime', 'afternoon', 'evening'])
    return df_copy


def feature_cyclical_hour(slot_df):
    """
    Creates cyclical features out of the hour_sched column.

    Args:
        slot_df: A row-per-appointment dataframe.

    Returns: A row-per-appointment dataframe with 2 additional columns: 'hour_sin' and 'hour_cos'.

    """

    df_copy = slot_df.copy()

    df_copy['hour_sin'] = np.sin(df_copy['hour_sched'] * (2. * np.pi / 24))
    df_copy['hour_cos'] = np.cos(df_copy['hour_sched'] * (2. * np.pi / 24))
    return df_copy


def feature_cyclical_day_of_week(slot_df):
    """
    Creates cyclical features out of the day_of_week column.

    Args:
        slot_df: A row-per-appointment dataframe.

    Returns: A row-per-appointment dataframe with 2 additional columns: 'day_of_week_sin' and 'day_of_weekcos'.

    """

    df_copy = slot_df.copy()

    df_copy['day_of_week_sin'] = np.sin(df_copy['day_of_week'] * (2. * np.pi / 5))
    df_copy['day_of_week_cos'] = np.cos(df_copy['day_of_week'] * (2. * np.pi / 5))
    return df_copy


def feature_cyclical_month(slot_df):
    """
    Creates cyclical features out of the month column.

    Args:
        slot_df: A row-per-appointment dataframe.

    Returns: A row-per-appointment dataframe with 2 additional columns: 'month_sin' and 'month_cos'.

    """

    df_copy = slot_df.copy()

    df_copy['month_sin'] = np.sin((df_copy['month'] - 1) * (2. * np.pi / 12))
    df_copy['month_cos'] = np.cos((df_copy['month'] - 1) * (2. * np.pi / 12))
    return df_copy


def feature_occupation(df):
    df_remap = df.copy()

    df_remap['occupation'] = ''
    df_remap['Beruf'] = df_remap['Beruf'].astype(str)
    df_remap['occupation'] = df_remap['occupation'].astype(str)

    df_remap.loc[df_remap['Beruf'] == 'nan', 'occupation'] = 'none_given'
    df_remap.loc[df_remap['Beruf'] == '-', 'occupation'] = 'none_given'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='rentner|Renter|pensioniert|pens.|rente'),
                 'occupation'] = 'retired'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='keine Angaben|keine Ang'),
                 'occupation'] = 'none_given'
    df_remap.loc[df_remap['Beruf'].apply(regex_search,
                                         search_str='Angestellte|ang.|baue|angest.|Hauswart|dozent|designer|^KV$|'
                                                    'masseu|Raumpflegerin|Apothekerin|Ing.|fotog|Psycholog|'
                                                    'Sozialpädagoge|Werkzeu|druck|musik|koordinator|software|'
                                                    'schaler|Kosmetikerin|Physiotherapeutin|Physiker|Unternehmer|'
                                                    'Praktikant|Analy|reinig|Detailhandel|putz|Grafiker|anwält|'
                                                    'maschinist|Immobilien|Zimmermann|schloss|Kassiererin|'
                                                    'hotel|hochbau|marketing|engineer|IT|Rechts|backer|bäcker|'
                                                    'baecker|Disponent|magazin|chemik|Journalist|Schreiner|metzg|'
                                                    'Consultant|Berater|Köch|gärtn|gartn|gaertn|Professor|'
                                                    'Praktikantin|Gipser|Küche|lehrl|logist|Buchhalter|technik|'
                                                    'Projektleiter|Manager|Assistent|Landwirt|Poliz|Elektro|'
                                                    'Elektri|Jurist|Kellner|Sekret|Lager|Monteur|Coiffeu|spengler|'
                                                    'Kindergärtner|Geschäfts|mechanik|maurer|Maler|Chauffeur|'
                                                    'ingenieur|Kauf|mitarbeiter|Verkäufer|Informatiker|koch|'
                                                    'lehrer|arbeiter|architekt'),
                 'occupation'] = 'employed'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='student|Schüler|Doktorand|'
                                                                  'Kind|Stud.|Ausbildung|^MA$'),
                 'occupation'] = 'student'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='^IV$|^IV-Bezüger|^$|arbeitslos|ohne Arbeit|'
                                                                  'ohne|o.A.|nicht Arbeitstätig|'
                                                                  'Sozialhilfeempfänger|o. Arbeit|keine Arbeit|'
                                                                  'Asyl|RAV|Hausfrau|Hausmann'),
                 'occupation'] = 'unemployed'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='selbst'), 'occupation'] = 'self_employed'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='arzt|aerzt|ärzt|pflegefachfrau|Pflegehelfer|'
                                                                  'MTRA|Erzieherin|Fachfrau Betreuung|'
                                                                  'Pflegefachmann|MPA|FaGe|Krankenschwester|'
                                                                  'Fachmann MTRA'),
                 'occupation'] = 'hospital_worker'
    df_remap.loc[df_remap['Beruf'].apply(regex_search, search_str='Tourist'), 'occupation'] = 'other'

    df_remap.loc[df_remap['occupation'] == '', 'occupation'] = 'other'
    df_remap.loc[df_remap['occupation'].isna(), 'occupation'] = 'other'
    df_remap = df_remap.drop('Beruf', axis=1)
    return df_remap


def feature_reason(status_df):
    df_remap = status_df.copy()

    df_remap['reason'] = 0
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search, search_str="verlauf"), 'reason'] = 'verlaufskontrolle'
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search, search_str="läsion|laesion"), 'reason'] = 'lesion'
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search, search_str="nachsorge"), 'reason'] = 'aftercare'
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search, search_str="ganzkörper"), 'reason'] = 'full_body'
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search, search_str="rezidiv"), 'reason'] = 'relapse_check'
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search,
                                                  search_str="entzündliche veränderungen|entzuendliche veraenderungen"),
                 'reason'] = 'entzuendliche_veraenderungen'
    df_remap.loc[df_remap['ReasonForStudy'].apply(regex_search, search_str="krebs|cancer|tumor|onkolog|HCC"),
                 'reason'] = 'cancer'
    df_remap.loc[df_remap['ReasonForStudy'] == 'nan', 'reason'] = 'none_given'
    df_remap.loc[df_remap['reason'] == 0, 'reason'] = 'other'
    return df_remap


# feature engineering for the duration model
def feature_duration(dicom_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the duration of each MRI examination in minutes.
    Returns: the original dataframe plus a duration column
    """

    dicom_df["duration"] = (dicom_df["image_end"] - dicom_df["image_start"]) / np.timedelta64(1, "m")
    return dicom_df


def regex_search(x, search_str):
    return bool(re.search(search_str, x, re.IGNORECASE))
