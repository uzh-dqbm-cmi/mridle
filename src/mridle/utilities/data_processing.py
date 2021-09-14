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
