import pandas as pd
import numpy as np
import datetime
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, prep_raw_df_for_parquet, build_slot_df
from mridle.pipelines.data_science.feature_engineering.nodes import remove_na, \
    generate_3_5_days_ahead_features, add_business_days, subtract_business_days, feature_no_show_before
from mridle.utilities.prediction import main as prediction_main
import os
import re
from dateutil.relativedelta import relativedelta

AGO_DIR = '/data/mridle/data/silent_live_test/live_files/all/ago/'
OUT_DIR = '/data/mridle/data/silent_live_test/live_files/all/out/'
PREDS_DIR = '/data/mridle/data/silent_live_test/live_files/all/out_features_data/'


def get_slt_status_data(ago_out, with_source_file_info=False):
    file_dir = '/data/mridle/data/silent_live_test/live_files/all/{}/'.format(ago_out)

    all_status = pd.DataFrame()

    for filename in os.listdir(file_dir):
        if filename.endswith(".csv"):
            f_status = pd.read_csv(os.path.join(file_dir, filename), encoding='utf-16')
            slt_df = f_status.copy()
            if with_source_file_info:
                slt_df['source_file'] = filename
        all_status = pd.concat([all_status, slt_df])

    rfs_df = pd.read_csv('/data/mridle/data/silent_live_test/live_files/all/'
                         'retrospective_reasonforstudy/content/[dbo].[MRIdle_retrospective].csv')
    rfs_df[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates()
    all_status = all_status.merge(rfs_df[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates(), on='FillerOrderNo',
                                  how='left')
    all_status['ReasonForStudy'] = all_status['ReasonForStudy_x'].fillna(all_status['ReasonForStudy_y'])
    all_status = all_status.drop(columns=['ReasonForStudy_x', 'ReasonForStudy_y'])

    all_status = all_status.drop_duplicates()
    all_status = prep_raw_df_for_parquet(all_status)
    all_status = build_status_df(all_status, exclude_patient_ids=[])
    return all_status


def get_slt_with_outcome():
    preds = pd.read_csv(
        '/data/mridle/data/silent_live_test/live_files/all/out_features_data/features_master_slt_features.csv',
        parse_dates=['start_time', 'end_time'])
    preds.drop(columns=['NoShow'], inplace=True)
    actuals = pd.read_csv('/data/mridle/data/silent_live_test/live_files/all/actuals/master_actuals_with_filename.csv',
                          parse_dates=['start_time', 'end_time'])

    slt_with_outcome = preds.merge(actuals[['start_time', 'MRNCmpdId', 'NoShow']], on=['start_time', 'MRNCmpdId'],
                                   how='left')
    slt_with_outcome['NoShow'].fillna(False, inplace=True)

    most_recent_actuals = np.max(actuals['start_time']).date()
    slt_with_outcome = slt_with_outcome[slt_with_outcome['start_time'].dt.date <= most_recent_actuals]
    return slt_with_outcome


def get_slt_features():
    file_dir = '/data/mridle/data/silent_live_test/live_files/all/out/'
    all_slt_features = pd.DataFrame()

    historical_data = pd.read_parquet('/data/mridle/data/kedro_data_catalog/04_feature/'
                                      'master_feature_set_na_removed.parquet')

    rfs_df = pd.read_csv('/data/mridle/data/silent_live_test/live_files/all/'
                         'retrospective_reasonforstudy/content/[dbo].[MRIdle_retrospective].csv')
    rfs_df[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates()

    # add on proper noshow
    ago_st = get_slt_status_data('ago')

    for filename in os.listdir(file_dir):
        if filename.endswith(".csv"):

            _, out_day, out_month, out_year = re.split('_', os.path.splitext(filename)[0])
            out_date_start = datetime.datetime(int(out_year), int(out_month), int(out_day))
            file_generation_date = subtract_business_days(out_date_start, 3)

            out_status = pd.read_csv(os.path.join(file_dir, filename), encoding='utf-16')

            out_status = out_status.merge(rfs_df[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates(),
                                          on='FillerOrderNo', how='left')

            if 'ReasonForStudy_x' in out_status.columns:
                out_status['ReasonForStudy'] = out_status['ReasonForStudy_x'].fillna(out_status['ReasonForStudy_y'])
                # Drop the columns no longer needed
                out_status.drop(columns=['ReasonForStudy_x', 'ReasonForStudy_y'], inplace=True)

            out_status = out_status.drop_duplicates()
            out_status = prep_raw_df_for_parquet(out_status)
            out_status = build_status_df(out_status, exclude_patient_ids=[])
            slt_data_features = generate_3_5_days_ahead_features(out_status, f_dt=file_generation_date)

        all_slt_features = pd.concat([all_slt_features, slt_data_features])

    all_slt_features.drop(columns=['NoShow'], inplace=True)
    actuals_end_dt = datetime.datetime.today() + relativedelta(months=1)

    actuals_data = build_slot_df(ago_st, valid_date_range=[pd.to_datetime('2022-02-01'), actuals_end_dt.date()])
    all_slt_features = all_slt_features.merge(actuals_data[['MRNCmpdId', 'start_time', 'NoShow']].drop_duplicates(),
                                              how='left', on=['MRNCmpdId', 'start_time'])
    all_slt_features['NoShow'].fillna(False, inplace=True)

    all_slt_features.drop(columns=['no_show_before', 'no_show_before_sq', 'appts_before',
                                   'show_before', 'no_show_rate'], inplace=True)
    # duplicates coming from no_show_before being removed - some patients have two rows, one with noshowbefore=0,
    # some with it equal to 1. LOOK INTO THAT
    all_slt_features = all_slt_features.drop_duplicates()

    all_slt_features = feature_no_show_before(all_slt_features, hist_data_df=historical_data)

    return all_slt_features


def get_sorted_filenames(file_dir):
    files_list = os.listdir(file_dir)
    relevant_files = [x for x in files_list if x.endswith(".csv")]
    splits = pd.DataFrame([re.split('_', os.path.splitext(x)[0]) for x in relevant_files])
    splits['filename'] = relevant_files
    splits.columns = ['type', 'day', 'month', 'year', 'filename']
    splits.sort_values(['year', 'month', 'day'], inplace=True)
    return splits


def process_live_data():

    # process_live_data() function
    already_processed_filename = '/data/mridle/data/silent_live_test/live_files/already_processed.txt'
    master_feature_set = pd.read_parquet(
        '/data/mridle/data/kedro_data_catalog/04_feature/master_feature_set_na_removed.parquet')

    rfs_file = "/data/mridle/data/silent_live_test/live_files/all/" \
               "retrospective_reasonforstudy/content/[dbo].[MRIdle_retrospective].csv"
    rfs = pd.read_csv(rfs_file)
    rfs = rfs[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates()
    rfs['ReasonForStudy'] = rfs['ReasonForStudy'].astype(str)

    with open(already_processed_filename, 'r') as f:
        already_processed_files = f.read().splitlines()

    with open('/data/mridle/data/silent_live_test/live_files/dates_to_ignore.txt', 'r') as f:
        dates_to_ignore = f.read().splitlines()

    ago_files = get_sorted_filenames(AGO_DIR)
    for idx, filename_row in ago_files.iterrows():
        if filename_row['filename'] not in dates_to_ignore and filename_row['filename'] not in already_processed_files:
            ago_day, ago_month, ago_year = filename_row[['day', 'month', 'year']]
            print(ago_day, ago_month, ago_year)
            filename = filename_row['filename']

            ago_date_start = datetime.datetime(int(ago_year), int(ago_month), int(ago_day))
            ago_date_end = ago_date_start
            ago_start_year, ago_start_month, ago_start_date = ago_date_start.strftime("%Y"), ago_date_start.strftime(
                "%m"), ago_date_start.strftime("%d")
            ago_end_year, ago_end_month, ago_end_date = ago_date_end.strftime("%Y"), ago_date_end.strftime(
                "%m"), ago_date_end.strftime("%d")

            ago_valid_date_range = ['{}-{}-{}'.format(ago_start_year, ago_start_month, ago_start_date),
                                    '{}-{}-{}'.format(ago_end_year, ago_end_month, ago_end_date)]
            ago = pd.read_csv(
                '/data/mridle/data/silent_live_test/live_files/all/ago/{}'.format(filename_row['filename']),
                encoding="utf_16")
            formatted_ago_df = prep_raw_df_for_parquet(ago)
            ago_status_df = build_status_df(formatted_ago_df, list())
            ago_status_df = ago_status_df.merge(rfs, how='left')
            ago_features_df_maybe_na = build_slot_df(ago_status_df, valid_date_range=ago_valid_date_range)
            ago_features_df = remove_na(ago_features_df_maybe_na)

            ago_features_df['file'] = filename

            master_ago_filepath = '/data/mridle/data/silent_live_test/live_files/all/' \
                                  'actuals/master_actuals_with_filename.csv'
            if os.path.exists(master_ago_filepath):
                master_ago = pd.read_csv(master_ago_filepath)
            else:
                master_ago = pd.DataFrame()

            master_ago_updated = pd.concat([master_ago, ago_features_df], axis=0)
            master_ago_updated.drop_duplicates(inplace=True)
            master_ago_updated.to_csv(master_ago_filepath, index=False)

            ago_features_df.to_csv(
                '/data/mridle/data/silent_live_test/live_files/all/actuals/actuals_{}_{}_{}_with_filename.csv'.format(
                    ago_day,
                    ago_month,
                    ago_year))

            with open(already_processed_filename, 'a') as ap_f:
                ap_f.write(f'\n{filename}')

    out_files = get_sorted_filenames(OUT_DIR)
    for idx, filename_row in out_files.iterrows():
        if filename_row['filename'] not in dates_to_ignore and filename_row['filename'] not in already_processed_files:
            filename = filename_row['filename']

            out_day, out_month, out_year = filename_row[['day', 'month', 'year']]
            print(filename_row['filename'], out_day, out_month, out_year)

            out_date_start = datetime.datetime(int(out_year), int(out_month), int(out_day))
            out_date_end = add_business_days(out_date_start, 2)

            out_start_year, out_start_month, out_start_date = out_date_start.strftime("%Y"), out_date_start.strftime(
                "%m"), out_date_start.strftime("%d")
            out_end_year, out_end_month, out_end_date = out_date_end.strftime("%Y"), out_date_end.strftime(
                "%m"), out_date_end.strftime("%d")

            out_valid_date_range = ['{}-{}-{}'.format(out_start_year, out_start_month, out_start_date),
                                    '{}-{}-{}'.format(out_end_year, out_end_month, out_end_date)]

            data_path = '/data/mridle/data/silent_live_test/live_files/all/out/{}'.format(filename_row['filename'])
            model_dir = '/data/mridle/data/kedro_data_catalog/06_models/'
            output_path = '/data/mridle/data/silent_live_test/live_files/all/' \
                          'out_features_data/features_{}_{}_{}.csv'.format(out_day, out_month, out_year)

            prediction_main(data_path, model_dir, output_path, valid_date_range=out_valid_date_range,
                            file_encoding='utf-16', master_feature_set=master_feature_set, rfs_df=rfs,
                            filename=filename)

            with open(already_processed_filename, 'a') as ap_f:
                ap_f.write(f'\n{filename}')


def get_silent_live_test_predictions(model_str='prediction_xgboost', all_columns=True):
    """
    We would provide names on Wednesday for the following Monday, Tuesday, Wednesday, and then on Monday for the coming
    Thursday and Friday. 60% of the names on the Wed, and 40% on the Monday.
    """
    i = 0
    preds_dir = '/data/mridle/data/silent_live_test/live_files/all/predictions/'
    for filename in os.listdir(preds_dir):
        if filename.endswith("2022.csv"):
            test_predictions = pd.read_csv(os.path.join(preds_dir, filename), parse_dates=['start_time'])

            preds = test_predictions.copy()
            preds.rename(columns={model_str: filename}, inplace=True)
            preds.drop(columns=[x for x in preds.columns if 'prediction_' in x], inplace=True)
            preds.drop(columns=[x for x in preds.columns if 'Unnamed:' in x], inplace=True)
            if i == 0:
                preds_merged = preds.copy()
                i = 1
            else:
                preds_merged = preds_merged.merge(preds, how='outer')

    pred_cols = [col for col in preds_merged.columns if 'preds' in col]
    preds_merged['prediction'] = preds_merged[pred_cols].bfill(axis=1).iloc[:, 0]
    if not all_columns:
        preds_merged = preds_merged[['start_time', 'MRNCmpdId', 'FillerOrderNo', 'prediction']]

    return preds_merged


def get_predictions_for_nurses(split_config=None, model_str='prediction_xgboost', all_columns=True):
    """
    We would provide names on Wednesday for the following Monday, Tuesday, Wednesday, and then on Monday for the coming
    Thursday and Friday. 60% of the names on the Wed, and 40% on the Monday.
    """
    if split_config is None:
        split_config = {
            'Monday': {
                'days': ['Monday', 'Tuesday', 'Wednesday'],
                'num_preds': 6
            },
            'Thursday': {
                'days': ['Thursday', 'Friday'],
                'num_preds': 4
            }
        }

    all_preds = pd.DataFrame()

    for filename in os.listdir(PREDS_DIR):
        if filename.endswith("2022.csv"):
            _, out_day, out_month, out_year = re.split('_', os.path.splitext(filename)[0])
            day_of_week_from_filename = datetime.datetime(int(out_year), int(out_month), int(out_day)).strftime('%A')

            if day_of_week_from_filename in split_config.keys():
                test_predictions = pd.read_csv(os.path.join(PREDS_DIR, filename), parse_dates=['start_time'])
                preds = test_predictions.copy()

                preds = preds[
                    preds['start_time'].dt.strftime('%A').isin(split_config[day_of_week_from_filename]['days'])]

                preds.rename(columns={model_str: "pred_{}".format(filename)}, inplace=True)
                preds.drop(columns=[x for x in preds.columns if 'prediction_' in x], inplace=True)
                preds.drop(columns=[x for x in preds.columns if 'Unnamed:' in x], inplace=True)

                preds = preds.sort_values("pred_{}".format(filename), ascending=False)[
                        :split_config[day_of_week_from_filename]['num_preds']]
                pred_cols = [col for col in preds.columns if 'preds' in col]
                preds['prediction'] = preds[pred_cols].bfill(axis=1).iloc[:, 0]
                preds.drop(columns=pred_cols, inplace=True)
                if not all_columns:
                    preds = preds[['start_time', 'MRNCmpdId', 'FillerOrderNo', 'prediction']]
                all_preds = pd.concat([all_preds, preds], axis=0)

    return all_preds


def get_silent_live_test_actuals(all_columns=True):
    all_actuals = pd.DataFrame()
    actuals_dir = '/data/mridle/data/silent_live_test/live_files/all/actuals/'
    i = 0
    for filename in os.listdir(actuals_dir):
        if filename.endswith("_2022.csv"):
            actuals = pd.read_csv(os.path.join(actuals_dir, filename), parse_dates=['start_time'])

            if not all_columns:
                actuals = actuals[['start_time', 'NoShow', 'MRNCmpdId', 'FillerOrderNo']]

            actuals['filename'] = filename

            if i == 0:
                all_actuals = actuals.copy()
                i = 1
            else:
                all_actuals = pd.concat([all_actuals, actuals], axis=0)
    return all_actuals
