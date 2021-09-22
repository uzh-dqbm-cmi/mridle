import numpy as np
import pandas as pd
import re
from typing import List


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