"""
Code to process the DICOM metadata extracts from the DFL-IT. This data extract comes with each appointment having
multiple rows associated with it, each row representing an image scanned by the MRI machine. We receive just a subset
of the metadata related to this image, and no actual image file.

The below functions are used in the data preprocessing step(s), and individual descriptions are given at the start
of each function.
"""

import datetime as dt
import numpy as np
import pandas as pd


def process_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take in the DICOM dataframe and add some date columns for easier calculations later

    Args:
        df: Dataframe where each row represents one image for an appointment,
         containing columns: ['AcquisitionDate', 'AcquisitionTime']

    Returns:
        dataframe with some reformatted date columns, and some new columns added
    """
    df_copy = df.copy()
    df_copy['AcquisitionDate'] = pd.to_datetime(df_copy['AcquisitionDate'], format='%Y%m%d').dt.date
    df_copy['AcquisitionTime'] = pd.to_datetime(df_copy['AcquisitionTime'], format='%H%M%S.%f').dt.time
    df_copy.loc[~df_copy['AcquisitionDate'].isnull(), 'acq_week'] = df_copy.loc[
        ~df_copy['AcquisitionDate'].isnull(), 'AcquisitionDate'].apply(lambda x: x.isocalendar().week)
    df_copy.loc[~df_copy['AcquisitionTime'].isnull(),
                'acq_datetime'] = df_copy[~df_copy['AcquisitionTime'].isnull()].apply(
        lambda x: dt.datetime.combine(x['AcquisitionDate'], x['AcquisitionTime']), axis=1)

    return df_copy


def remove_na_and_duplicates(df: pd.DataFrame):
    """
    Removes duplicate rows and removes appointments with no AccessionNumber (appointment ID) or no data given

    TODO: Determine the cause of Null AccessionNumbers

    Args:
        df: Dataframe containing columns ['AccessionNumber', 'AcquisitionDate']
    Returns:
        Dataframe with duplicates removed and rows with null AccessionNumber or AcquisitionData columns

    """
    df_copy = df.copy()
    df_copy = df_copy.drop_duplicates()
    df_copy = df_copy[~df_copy['AccessionNumber'].isna()]
    df_copy = df_copy[~df_copy['AcquisitionDate'].isnull()]

    return df_copy


def add_image_time_cols(df):
    """
    Adds columns describing the order (rank) and time differences between images within an appointment to the dataframe.

    The first group of these columns is time information for the previous and the next image in the sequence of images
    per appointment. Also added are columns calculating the time between the previous image in the appointment and the
    row image, as well as the time between the row image and the next image. (row image meaning the image that the
    given row is related to). Finally for this group, a flag is added for each image in the appointment if two images
    in the sequence are more than 1800 seconds / 30 minutes apart.

    The second group of columns is the 'rank' of the image, or what position it is in the sequence of images for the
    appointment (i.e. is it the first image, second, ..., 20th, ..., second last, last). And the same column idea
    is provided in a separate column, but with the rank in reverse order - this is mostly for easier calculations later
    and to make it clear which images are really the last (rather than just showing that they are the 254th image in a
    sequence of 254 images).

    Args:
        df: Dataframe where each row represents one image for an appointment,
         containing columns: ['AccessionNumber', 'AcquisitionTime', 'acq_datetime']

    Returns:
        Dataframe with the columns as described above added.

    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(['AccessionNumber', 'acq_datetime'])
    df_copy['AcquisitionTime_prev'] = df_copy.groupby('AccessionNumber')['AcquisitionTime'].shift(1)
    df_copy['acq_prev_datetime'] = df_copy.groupby('AccessionNumber')['acq_datetime'].shift(1)
    df_copy['acq_next_datetime'] = df_copy.groupby('AccessionNumber')['acq_datetime'].shift(-1)

    one_second = pd.to_timedelta(1, unit='S')
    df_copy['time_between_next_image'] = (df_copy['acq_next_datetime'] - df_copy['acq_datetime']) / one_second
    df_copy['time_between_prev_image'] = (df_copy['acq_datetime'] - df_copy['acq_prev_datetime']) / one_second

    df_copy['big_image_gap'] = np.max(df_copy[['time_between_next_image', 'time_between_prev_image']], axis=1)
    df_copy['big_image_gap'] = df_copy[['AccessionNumber', 'big_image_gap']].groupby('AccessionNumber').transform(
        lambda x: 1 if np.max(x) > 1800 else 0)

    df_copy['img_rank'] = df_copy.groupby('AccessionNumber')['acq_datetime'].transform('rank', ascending=True)
    df_copy['img_rank_rev'] = df_copy.groupby('AccessionNumber')['acq_datetime'].transform('rank', ascending=False)

    return df_copy


def remove_gaps_at_start_end(df):
    """
    If there is a large gap (default of 30 minutes, 1800 seconds) contained within the first or last 5 images of the
    sequence, remove the images before/after this gap (before if it's at the start of an sequence, after if it's at
    the end).

    e.g. if there are 3 images taken at ~9am, and then a gap of 90 minutes until the remaining images are taken as part
    of this AccessionNumber (could be 100+ images), we remove these 3 images at the start, leaving behind only the
    'true' section of the appointment. It is assumed that these original 3 images were created as a computer error,
    and are not genuinely relating to the appointment.

    - Requires columns that are added by the get_image_time_cols() function

    Args:
        df: Dataframe with multiple rows per AccessionNumber (each row is the metadata for an individual image)

    Returns:
        Dataframe with rows removed as necessary.

    """
    df_copy = df.copy()
    remove_before = df_copy.loc[(df_copy['img_rank'] <= 5) & (df_copy['time_between_next_image'] > 1800),
                                w["AccessionNumber", "img_rank"]]
    remove_after = df_copy.loc[(df_copy['img_rank_rev'] <= 5) & (df_copy['time_between_prev_image'] > 1800),
                               ["AccessionNumber", "img_rank_rev"]]

    for idx, row in remove_before.iterrows():
        df_copy = df_copy[~((df_copy['img_rank'] <= row['img_rank']) &
                            (df_copy['AccessionNumber'] == row['AccessionNumber']))]

    for idx, row in remove_after.iterrows():
        df_copy = df_copy[~((df_copy['img_rank_rev'] <= row['img_rank_rev']) &
                            (df_copy['AccessionNumber'] == row['AccessionNumber']))]

    return df_copy
