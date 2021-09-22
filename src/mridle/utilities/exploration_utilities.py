import pandas as pd


def view_status_changes(df: pd.DataFrame, fon: int) -> pd.DataFrame:
    """
    View the status changes for an appointment.
    Args:
        df: row-per-status-change df.
        fon: FillerOrderNo

    Returns:

    """
    row = df[df['FillerOrderNo'] == fon].copy()
    return row.sort_values('History_MessageDtTm')


def view_status_changes_of_random_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    View the status changes for an appointment.
    Args:
        df: row-per-status-change df.

    Returns:

    """
    row = df[df['PatientClass'] == 'ambulant'].sample(1)
    fon = row['FillerOrderNo'].iloc[0]
    return view_status_changes(df, fon)
