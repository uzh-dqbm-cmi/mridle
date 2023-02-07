import pandas as pd
import numpy as np


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
