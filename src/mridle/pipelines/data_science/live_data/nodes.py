import pandas as pd
import numpy as np
import datetime


def get_slt_with_outcome():
    preds = pd.read_csv(
        '/data/mridle/data/silent_live_test/live_files/all/out_features_data/features_master_slt_features.csv',
        parse_dates=['start_time', 'end_time'])
    preds.drop(columns=['NoShow'], inplace=True)
    actuals = pd.read_csv('/data/mridle/data/silent_live_test/live_files/all/actuals/master_actuals_with_filename.csv',
                          parse_dates=['start_time', 'end_time'])

    preds['MRNCmpdId'] = preds['MRNCmpdId'].astype(str)
    actuals['MRNCmpdId'] = actuals['MRNCmpdId'].astype(str)

    slt_with_outcome = preds.merge(actuals[['start_time', 'MRNCmpdId', 'NoShow']], on=['start_time', 'MRNCmpdId'],
                                   how='left')
    slt_with_outcome['NoShow'].fillna(False, inplace=True)

    most_recent_actuals = np.max(actuals['start_time'])  # .date()
    slt_with_outcome = slt_with_outcome[slt_with_outcome['start_time'] <= most_recent_actuals]
    return slt_with_outcome


def concat_master_data(master_feature_set_na_removed, live_data):
    """Take live data up until start of last month, and concat with master feature set. That is then training data.
    Rest of live data (i.e. from start of last month until now) is then validation data"""
    for col in live_data.columns:
        live_data[col] = live_data[col].astype(master_feature_set_na_removed[col].dtypes.name)

    last_monday = datetime.date.today() + datetime.timedelta(days=-datetime.date.today().weekday())
    five_weeks_ago = last_monday - datetime.timedelta(weeks=5)

    live_data_train = live_data[live_data['start_time'].dt.date < five_weeks_ago]
    val_data_with_live = live_data[live_data['start_time'].dt.date >= five_weeks_ago]

    train_data_with_live = pd.concat([master_feature_set_na_removed, live_data_train], join="inner")

    return train_data_with_live, val_data_with_live
