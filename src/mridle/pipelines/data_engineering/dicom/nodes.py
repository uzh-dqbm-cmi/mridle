"""
Code to process the DICOM metadata extracts from the DFL-IT. This data extract comes with each appointment having
multiple rows associated with it, each row representing an image scanned by the MRI machine. We receive just a subset
of the metadata related to this image, and no actual image file.

The below functions are used in the data preprocessing step(s), and individual descriptions are given at the start
of each function.
"""
import pandas as pd
import datetime as dt
import numpy as np


def preprocess_dicom_data(df, id_list_df):
    dicom_5_years = df.copy()
    dicom_5_years = subset_valid_appts(dicom_5_years, id_list_df)
    dicom_5_years = subset_machines(dicom_5_years)
    dicom_5_years = remove_na_and_duplicates(dicom_5_years)
    dicom_5_years = process_date_cols(dicom_5_years)
    dicom_5_years = add_image_time_cols(dicom_5_years)
    dicom_5_years = remove_gaps_at_start_end(dicom_5_years)
    # dicom_5_years = add_image_time_cols(dicom_5_years)
    return dicom_5_years


def aggregate_dicom_images(df):
    df_copy = df.copy()
    df_copy_agg = df_copy.groupby(['AccessionNumber', 'big_image_gap', 'StationName']).agg(
        {'acq_datetime': [min, max]}).reset_index()
    df_copy_agg.columns = ['AccessionNumber', 'big_image_gap', 'image_device_id', 'image_start', 'image_end']
    dicom_data = df_copy_agg[['AccessionNumber', 'image_device_id', 'image_start', 'image_end']]
    return dicom_data


def integrate_dicom_data(slot_df: pd.DataFrame, dicom_times_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate DICOM data into slot_df to update the appointment information with more reliable information.
    - For all appointments, the device_id is updated with the one listed in dicom_times_df (the device actually used, as
        opposed to the device that was planned to be used).
    - For show and inpatient appointments, the appointment start and end times are updated to the accurate DICOM values
        (the actual imaging start and end times, as opposed to the planned appt start time).
    Original values from the status_df are maintained in separate columns: status_start, status_end, device_from_status.
    Args:
        slot_df: row-per-status-change dataframe.
        dicom_times_df: A dataframe of DICOM metadata information, with the columns:
            - AccessionNumber
            - image_device_id
            - image_start
            - image_end
    Returns: slot_df with new and modified time and device columns. Columns include:
        - start_time (as determined by DICOM data)
        - status_start (start time as originally stated by status data)
        - end_time (as determined by DICOM data)
        - status_end (end time as originally stated by status data)
        - EnteringOrganisationDeviceID (device as determined by DICOM data)
        - device_from_status (device as originally stated by status data)
    Raises:
        ValueError if number of rows in status_df changes during this transformation
    """
    slot_w_dicom_df = pd.merge(slot_df, dicom_times_df, how='left', left_on='FillerOrderNo', right_on='AccessionNumber')

    # move times defined by status changes to separate columns to allow overwriting the original columns with dicom data
    slot_w_dicom_df['status_start'] = slot_w_dicom_df['start_time']
    slot_w_dicom_df['status_end'] = slot_w_dicom_df['end_time']

    # for show and in-patient appointments, use dicom data for start and end times
    slot_w_dicom_df['start_time'] = slot_w_dicom_df.apply(update_start_time_col_from_dicom, axis=1)
    slot_w_dicom_df['end_time'] = slot_w_dicom_df.apply(update_end_time_col_from_dicom, axis=1)

    # update device used
    slot_w_dicom_df['device_from_status'] = slot_w_dicom_df['EnteringOrganisationDeviceID']
    slot_w_dicom_df['EnteringOrganisationDeviceID'] = slot_w_dicom_df.apply(update_device_id_from_dicom, axis=1)

    if slot_df.shape[0] != slot_w_dicom_df.shape[0]:
        raise ValueError('Number of rows in slot_w_dicom_df ({:,.0f}) does not match original slot_df ({:,.0f})'.format(
            slot_w_dicom_df.shape[0], slot_df.shape[0]))

    return slot_w_dicom_df


# Helper functions
def subset_valid_appts(df, id_list_df):
    df_copy = df.copy()
    df_copy['AccessionNumber'] = pd.to_numeric(df_copy['AccessionNumber'], errors='coerce')
    df_copy = df_copy[~df_copy['AccessionNumber'].isna()]

    id_list_df = id_list_df.drop_duplicates()
    id_list_df_pivot = id_list_df.pivot(['SeriesUID', 'SopImageUid', 'StudyUID'], columns='tag_name',
                                        values='Value').reset_index()
    id_list_df_pivot['AcquisitionDate'] = id_list_df_pivot['AcquisitionDate'].astype(float)
    id_list_df_pivot['AccessionNumber'] = pd.to_numeric(id_list_df_pivot['AccessionNumber'], errors='coerce')
    id_list_df_pivot = id_list_df_pivot[~id_list_df_pivot['AccessionNumber'].isna()]

    id_list = id_list_df_pivot['AccessionNumber'].unique()
    df_copy = df_copy[df_copy['AccessionNumber'].isin(id_list)]
    return df_copy


def subset_machines(df):
    df_copy = df.copy()
    df_copy = df_copy[df_copy['StationName'].isin(['MT00000173', 'MT00000213'])]
    df_copy['StationName'] = df_copy['StationName'].map({'MT00000173': '1', 'MT00000213': '2'})
    return df_copy


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

    df_copy['AcquisitionTime'] = df_copy['AcquisitionTime'].apply(lambda a: "{:013.6F}".format(float(a)))
    df_copy['AcquisitionDate'] = pd.to_datetime(df_copy['AcquisitionDate'], format='%Y%m%d').dt.date
    df_copy['AcquisitionTime'] = pd.to_datetime(df_copy['AcquisitionTime'], format='%H%M%S.%f').dt.time
    df_copy.loc[~df_copy['AcquisitionTime'].isnull(),
                'acq_datetime'] = df_copy[~df_copy['AcquisitionTime'].isnull()].apply(
        lambda x: dt.datetime.combine(x['AcquisitionDate'], x['AcquisitionTime']), axis=1)

    return df_copy


def remove_na_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
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


def add_image_time_cols(df: pd.DataFrame) -> pd.DataFrame:
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


def remove_gaps_at_start_end(df: pd.DataFrame) -> pd.DataFrame:
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
                                ["AccessionNumber", "img_rank"]]
    remove_after = df_copy.loc[(df_copy['img_rank_rev'] <= 5) & (df_copy['time_between_prev_image'] > 1800),
                               ["AccessionNumber", "img_rank_rev"]]

    for idx, row in remove_before.iterrows():
        df_copy = df_copy[~((df_copy['img_rank'] <= row['img_rank']) &
                            (df_copy['AccessionNumber'] == row['AccessionNumber']))]

    for idx, row in remove_after.iterrows():
        df_copy = df_copy[~((df_copy['img_rank_rev'] <= row['img_rank_rev']) &
                            (df_copy['AccessionNumber'] == row['AccessionNumber']))]

    return df_copy


def update_start_time_col_from_dicom(row):
    if row['slot_type'] in ['show', 'inpatient'] and row['image_start'] is not None:
        return row['image_start']
    return row['status_start']


def update_end_time_col_from_dicom(row):
    if row['slot_type'] in ['show', 'inpatient'] and row['image_end'] is not None:
        return row['image_end']
    return row['status_end']


def update_device_id_from_dicom(row):
    if pd.isna(row['image_device_id']):
        return row['EnteringOrganisationDeviceID']
    else:
        return 'MR{}'.format(int(row['image_device_id']))
