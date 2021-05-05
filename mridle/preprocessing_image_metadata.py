import datetime as dt
import numpy as np
import pandas as pd


def clean_marc_extract(df):
    # df = df[['AccessionNumber', 'AcquisitionDate', 'AcquisitionTime', 'DeviceSerialNumber', 'StationName', 'SeriesUID', 'StudyUID']].copy()
    df['AcquisitionDate'] = pd.to_datetime(df['AcquisitionDate'], format='%Y%m%d').dt.date
    df['AcquisitionTime'] = pd.to_datetime(df['AcquisitionTime'], format='%H%M%S.%f').dt.time
    df.loc[~df['AcquisitionDate'].isnull(), 'acq_week'] = df.loc[
        ~df['AcquisitionDate'].isnull(), 'AcquisitionDate'].apply(lambda x: x.isocalendar().week)
    df = df.drop_duplicates()
    df = df[~df['AccessionNumber'].isna()]
    df = df[~df['AcquisitionDate'].isnull()]
    df.loc[~df['AcquisitionTime'].isnull(), 'acq_datetime'] = df[~df['AcquisitionTime'].isnull()].apply(
        lambda x: dt.datetime.combine(x['AcquisitionDate'], x['AcquisitionTime']), axis=1)

    return df


def get_image_time_cols(df):
    df = df.sort_values(['AccessionNumber', 'acq_datetime'])
    df['AcquisitionTime_prev'] = df.groupby('AccessionNumber')['AcquisitionTime'].shift(1)
    df['acq_prev_datetime'] = df.groupby('AccessionNumber')['acq_datetime'].shift(1)
    df['acq_next_datetime'] = df.groupby('AccessionNumber')['acq_datetime'].shift(-1)

    df['img_rank'] = df.groupby('AccessionNumber')['acq_datetime'].transform('rank', ascending=True)
    df['img_rank_rev'] = df.groupby('AccessionNumber')['acq_datetime'].transform('rank', ascending=False)

    df['time_between_next_image'] = (df['acq_next_datetime'] - df['acq_datetime']) / pd.to_timedelta(1, unit='S')
    df['time_between_prev_image'] = (df['acq_datetime'] - df['acq_prev_datetime']) / pd.to_timedelta(1, unit='S')

    df['big_image_gap'] = np.max(df[['time_between_next_image', 'time_between_prev_image']], axis=1)
    df['big_image_gap'] = df[['AccessionNumber', 'big_image_gap']].groupby('AccessionNumber').transform(
        lambda x: 1 if np.max(x) > 1800 else 0)

    return df


def remove_start_end_images(df, longer_than_secs=1800, within_first_last=5):
    remove_before = df.loc[(df['img_rank'] <= within_first_last) &
                           (df['time_between_next_image'] > longer_than_secs), ["AccessionNumber", "img_rank"]]
    remove_after = df.loc[(df['img_rank_rev'] <= within_first_last) &
                          (df['time_between_prev_image'] > longer_than_secs), ["AccessionNumber", "img_rank_rev"]]

    for idx, row in remove_before.iterrows():
        df = df[~((df['img_rank'] <= row['img_rank']) & (df['AccessionNumber'] == row['AccessionNumber']))]

    for idx, row in remove_after.iterrows():
        df = df[~((df['img_rank_rev'] <= row['img_rank_rev']) & (df['AccessionNumber'] == row['AccessionNumber']))]

    return df
