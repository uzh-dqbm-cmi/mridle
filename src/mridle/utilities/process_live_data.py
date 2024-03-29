import pandas as pd
import datetime
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, prep_raw_df_for_parquet, build_slot_df
from mridle.pipelines.data_science.feature_engineering.nodes import remove_na, \
    generate_3_5_days_ahead_features, add_business_days, subtract_business_days, feature_no_show_before
from mridle.pipelines.data_science.live_data.nodes import get_slt_with_outcome
from mridle.experiment.experiment import Experiment
from mridle.experiment.dataset import DataSet
import os
import re
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pickle
import numpy as np
import csv

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


def get_slt_features_delete_if_ok_to_do_so():
    file_dir = '/data/mridle/data/silent_live_test/live_files/all/out/'
    all_slt_features = pd.DataFrame()

    historical_data = pd.read_parquet('/data/mridle/data/kedro_data_catalog/04_feature/'
                                      'master_feature_set_na_removed.parquet')

    rfs_df = pd.read_csv('/data/mridle/data/silent_live_test/live_files/all/'
                         'retrospective_reasonforstudy/content/[dbo].[MRIdle_retrospective].csv')
    rfs_df[['FillerOrderNo', 'ReasonForStudy']].drop_duplicates(inplace=True)

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

            ago_features_df = ago_features_df[ago_features_df['MRNCmpdId'].str[:3] != 'SMS']

            ago_features_df['file'] = filename

            master_ago_filepath = '/data/mridle/data/silent_live_test/live_files/all/' \
                                  'actuals/master_actuals.csv'
            if os.path.exists(master_ago_filepath):
                master_ago = pd.read_csv(master_ago_filepath)
            else:
                master_ago = pd.DataFrame()

            master_ago_updated = pd.concat([master_ago, ago_features_df], axis=0)
            master_ago_updated.drop_duplicates(inplace=True)
            master_ago_updated.to_csv(master_ago_filepath, index=False)

            ago_features_df.to_csv(
                '/data/mridle/data/silent_live_test/live_files/all/actuals/actuals_{}_{}_{}.csv'.format(
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
                          'out_features_data/features_{}_{}_{}.csv'.format(out_year, out_month, out_day)

            make_out_prediction(data_path, model_dir, output_path, valid_date_range=out_valid_date_range,
                                file_encoding='utf-16', master_feature_set=master_feature_set, rfs_df=rfs,
                                filename=filename)

            with open(already_processed_filename, 'a') as ap_f:
                ap_f.write(f'\n{filename}')


def make_out_prediction(data_path, model_dir, output_path, valid_date_range, file_encoding, master_feature_set, rfs_df,
                        filename):
    """
    Make predictions for all models in model_dir on the given data, saving the resulting predictions to output_path.
    Args:
        data_path: Path to input data (csv).
        model_dir: Directory to the model experiments. This directory should contain a set of directories containing
         pickled serialized `Experiments`.
        output_path: Destination to save the predictions file to (csv).
        valid_date_range: Date range on which to filter slot_df for relevant appointment slots.
        file_encoding: Encoding for file in pd.read_csv()

    Returns: None
    """

    def remove_redundant(df):
        """status 'changes' that aren't actually changes, that can often cause issues...let's remove these"""
        st_df = df.copy()
        st_df = st_df[~((st_df['now_status'] == st_df['was_status'])
                        & (st_df['now_sched_for_date'] == st_df['was_sched_for_date']))]
        return st_df

    try:
        raw_df = pd.read_csv(data_path, encoding=file_encoding)
    except pd.errors.ParserError:
        fix_csv_file(data_path)
        raw_df = pd.read_csv(data_path, encoding=file_encoding)

    exclude_pat_ids = list()  # TODO!

    start_dt = pd.to_datetime(valid_date_range[0])
    end_dt = pd.to_datetime(valid_date_range[1])

    formatted_df = prep_raw_df_for_parquet(raw_df)
    status_df = build_status_df(formatted_df, exclude_pat_ids)
    status_df = status_df.merge(rfs_df, how='left')
    status_df = remove_redundant(status_df)

    # Remove appts where last status is 'canceled'
    status_df = status_df.rename_axis('idx').sort_values('idx')
    last_status = status_df.rename_axis('idx').groupby(['FillerOrderNo']).apply(
        lambda x: x.sort_values(['History_MessageDtTm', 'idx'], ascending=[True, True]).head(1)
    ).reset_index(drop=True)[['MRNCmpdId', 'FillerOrderNo', 'now_status', 'now_sched_for_busday']]
    fon_to_remove = last_status.loc[(last_status['now_status'] == 'canceled') &
                                    (last_status['now_sched_for_busday'] > 3),
                                    'FillerOrderNo']

    status_df = status_df[~status_df['FillerOrderNo'].isin(fon_to_remove)]

    features_df_maybe_na = generate_3_5_days_ahead_features(status_df, start_dt, live_data=True)
    features_df = remove_na(features_df_maybe_na)

    features_df = features_df[features_df['MRNCmpdId'].str[:3] != 'SMS']

    day_after_last_valid_date = end_dt + pd.to_timedelta(1, 'days')
    features_df = features_df[features_df['start_time'] >= start_dt]
    features_df = features_df[features_df['start_time'] < day_after_last_valid_date]

    # Get number of previous no shows from historical data and add to data set
    master_df = master_feature_set.copy()
    master_df = master_df[master_df['MRNCmpdId'] != 'SMS0016578']

    master_slt_feature_filepath = '/data/mridle/data/silent_live_test/live_files/all/' \
                                  'out_features_data/features_master_slt_features.csv'

    if os.path.exists(master_slt_feature_filepath):
        master_slt_with_outcome = get_slt_with_outcome()
        master_feature_slt = pd.read_csv(master_slt_feature_filepath, parse_dates=['start_time'])
    else:
        master_slt_with_outcome = pd.DataFrame()
        master_feature_slt = pd.DataFrame()

    historic_data = pd.concat([master_df, master_slt_with_outcome], axis=0)

    historic_data['MRNCmpdId'] = historic_data['MRNCmpdId'].astype(str)
    features_df['MRNCmpdId'] = features_df['MRNCmpdId'].astype(str)

    hist_no_dupe = historic_data.merge(features_df[['FillerOrderNo', 'start_time']], how='left', indicator=True)
    hist_no_dupe = hist_no_dupe[hist_no_dupe['_merge'] == 'left_only']

    features_df = feature_no_show_before(features_df, hist_no_dupe)
    features_df['filename'] = filename

    model_dirs = Path(model_dir).glob('*')
    for model_dir in model_dirs:
        if str(model_dir) in ['/data/mridle/data/kedro_data_catalog/06_models/xgboost',
                              '/data/mridle/data/kedro_data_catalog/06_models/random_forest',
                              '/data/mridle/data/kedro_data_catalog/06_models/logistic_regression']:
            model_paths = model_dir.glob('*')
            for model_path in model_paths:
                with open(model_path, "rb+") as f:
                    serialized_model = pickle.load(f)
                exp = Experiment.deserialize(serialized_model)
                data_set = DataSet(exp.stratified_dataset.config, features_df)
                preds_proba = exp.final_predictor.predict_proba(data_set.x)
                model_name = exp.metadata.get('name', model_path.name)
                features_df[f'prediction_{model_name}'] = preds_proba

    features_df.to_csv(output_path, index=False)
    print(features_df.shape)

    new_appts = features_df.merge(historic_data[['FillerOrderNo', 'start_time']], how='left', indicator=True)
    new_appts = new_appts[new_appts['_merge'] == 'left_only']
    new_appts.drop(columns=['_merge'], inplace=True)
    print(new_appts.shape)
    # print(features_df[features_df['MRNCmpdId'] == '60184934'][['MRNCmpdId', 'FillerOrderNo', 'no_show_before',
    #                                                            'start_time', 'no_show_rate', 'NoShow']])
    master_slt_updated = pd.concat([master_feature_slt, new_appts], axis=0)
    master_slt_updated.drop_duplicates(inplace=True)
    master_slt_updated.to_csv(master_slt_feature_filepath, index=False)


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


def fix_csv_file(filename_to_fix):

    res = []

    with open(filename_to_fix, 'r', encoding='utf-16') as read_obj:
        # pass the file object to reader() to get the reader object
        # csv_reader = reader(read_obj, skipinitialspace=True)
        csv_reader = csv.DictReader(read_obj, restkey='ReasonForStudy2')
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            res.append(row)

    res_df = pd.DataFrame(res)
    if 'ReasonForStudy2' in res_df.columns:
        res_df['ReasonForStudy'] = np.where(res_df['ReasonForStudy2'].isna(), res_df['ReasonForStudy'],
                                            res_df['ReasonForStudy'].astype(str) + res_df['ReasonForStudy2'].astype(
                                                str))
        res_df.drop(columns=['ReasonForStudy2'], inplace=True)
        res_df['ReasonForStudy'].replace('"|,', " ", inplace=True)

    res_df.to_csv(filename_to_fix, encoding='utf-16')
    return None
