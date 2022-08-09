import pandas as pd
import numpy as np
import datetime
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, build_slot_df, prep_raw_df_for_parquet
from mridle.pipelines.data_science.feature_engineering.nodes import build_model_data, remove_na
from mridle.utilities.prediction import main as prediction_main
import os
import re


def add_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5:  # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date


def subtract_business_days(from_date, subtract_days):
    business_days_to_subtract = subtract_days
    current_date = from_date
    while business_days_to_subtract > 0:
        current_date -= datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5:  # sunday = 6
            continue
        business_days_to_subtract -= 1
    return current_date


def main():

    ago_dir = '/data/mridle/data/silent_live_test/live_files/all/ago/'
    already_processed_filename = '/data/mridle/data/silent_live_test/live_files/already_processed.txt'
    master_feature_set = pd.read_parquet(
        '/data/mridle/data/kedro_data_catalog/04_feature/master_feature_set_na_removed.parquet')
    previous_no_shows = master_feature_set[['MRNCmpdId', 'no_show_before']].groupby(['MRNCmpdId']).apply(
        np.max).reset_index(drop=True)
    rfs_file = "/data/mridle/data/silent_live_test/live_files/all/" \
               "retrospective_reasonforstudy/content/[dbo].[MRIdle_retrospective].csv"
    rfs = pd.read_csv(rfs_file)
    rfs = rfs[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates()
    rfs['ReasonForStudy'] = rfs['ReasonForStudy'].astype(str)

    with open(already_processed_filename, 'r') as f:
        already_processed_files = f.read().splitlines()

    with open('/data/mridle/data/silent_live_test/live_files/dates_to_ignore.txt', 'r') as f:
        dates_to_ignore = f.read().splitlines()

    for filename in os.listdir(ago_dir):
        if filename.endswith(".csv") and filename not in dates_to_ignore and filename not in already_processed_files:

            _, ago_day, ago_month, ago_year = re.split('_', os.path.splitext(filename)[0])
            ago_date_start = datetime.datetime(int(ago_year), int(ago_month), int(ago_day))
            ago_date_end = ago_date_start
            ago_start_year, ago_start_month, ago_start_date = ago_date_start.strftime("%Y"), ago_date_start.strftime(
                "%m"), ago_date_start.strftime("%d")
            ago_end_year, ago_end_month, ago_end_date = ago_date_end.strftime("%Y"), ago_date_end.strftime(
                "%m"), ago_date_end.strftime("%d")

            ago_valid_date_range = ['{}-{}-{}'.format(ago_start_year, ago_start_month, ago_start_date),
                                    '{}-{}-{}'.format(ago_end_year, ago_end_month, ago_end_date)]
            ago = pd.read_csv('/data/mridle/data/silent_live_test/live_files/all/ago/{}'.format(filename),
                              encoding="utf_16")
            formatted_ago_df = prep_raw_df_for_parquet(ago)
            ago_status_df = build_status_df(formatted_ago_df, list())
            ago_status_df = ago_status_df.merge(rfs, how='left')
            ago_slot_df = build_slot_df(ago_status_df, valid_date_range=ago_valid_date_range)
            ago_features_df_maybe_na = build_model_data(ago_status_df, valid_date_range=ago_valid_date_range,
                                                        slot_df=ago_slot_df)
            ago_features_df = remove_na(ago_features_df_maybe_na)
            ago_features_df = ago_features_df.merge(previous_no_shows, on='MRNCmpdId', how='left',
                                                    suffixes=['_current', '_hist'])
            ago_features_df['no_show_before_hist'].fillna(0, inplace=True)
            ago_features_df['no_show_before'] = ago_features_df['no_show_before_current'] + ago_features_df[
                'no_show_before_hist']
            ago_features_df.drop(['no_show_before_current', 'no_show_before_hist'], axis=1, inplace=True)
            ago_features_df['no_show_before_sq'] = ago_features_df['no_show_before'] ** 2
            ago_features_df.to_csv(
                '/data/mridle/data/silent_live_test/live_files/all/actuals/actuals_{}_{}_{}_test.csv'.format(ago_day,
                                                                                                             ago_month,
                                                                                                             ago_year))
            with open(already_processed_filename, 'a') as ap_f:
                ap_f.write(f'\n{filename}')

    out_dir = '/data/mridle/data/silent_live_test/live_files/all/out/'
    for filename in os.listdir(out_dir):
        if filename.endswith(".csv") and filename not in dates_to_ignore and filename not in already_processed_files:
            _, out_day, out_month, out_year = re.split('_', os.path.splitext(filename)[0])
            print(filename)
            out_date_start = datetime.datetime(int(out_year), int(out_month), int(out_day))
            out_date_end = add_business_days(out_date_start, 2)

            out_start_year, out_start_month, out_start_date = out_date_start.strftime("%Y"), out_date_start.strftime(
                "%m"), out_date_start.strftime("%d")
            out_end_year, out_end_month, out_end_date = out_date_end.strftime("%Y"), out_date_end.strftime(
                "%m"), out_date_end.strftime("%d")

            out_valid_date_range = ['{}-{}-{}'.format(out_start_year, out_start_month, out_start_date),
                                    '{}-{}-{}'.format(out_end_year, out_end_month, out_end_date)]

            data_path = '/data/mridle/data/silent_live_test/live_files/all/out/{}'.format(filename)
            model_dir = '/data/mridle/data/kedro_data_catalog/06_models/'
            output_path = '/data/mridle/data/silent_live_test/live_files/all/' \
                          'predictions/preds_{}_{}_{}_test.csv'.format(out_day, out_month, out_year)

            prediction_main(data_path, model_dir, output_path, valid_date_range=out_valid_date_range,
                            file_encoding='utf-16', master_feature_set=master_feature_set, rfs_df=rfs)

            with open(already_processed_filename, 'a') as ap_f:
                ap_f.write(f'\n{filename}')
