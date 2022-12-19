import argparse
import pandas as pd
from pathlib import Path
import pickle
import os

from mridle.pipelines.data_engineering.ris.nodes import build_status_df, prep_raw_df_for_parquet
from mridle.pipelines.data_science.feature_engineering.nodes import remove_na, generate_training_data, \
    feature_no_show_before
from mridle.experiment.experiment import Experiment
from mridle.experiment.dataset import DataSet
from mridle.utilities.process_live_data import get_slt_with_outcome


def main(data_path, model_dir, output_path, valid_date_range, file_encoding, master_feature_set, rfs_df, filename):
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

    if file_encoding:
        raw_df = pd.read_csv(data_path, encoding=file_encoding)
    else:
        raw_df = pd.read_csv(data_path)

    exclude_pat_ids = list()  # TODO!

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

    features_df_maybe_na = generate_training_data(status_df, valid_date_range, append_outcome=False,
                                                  add_no_show_before=False)
    features_df = remove_na(features_df_maybe_na)

    # Get number of previous no shows from historical data and add to data set
    master_df = master_feature_set.copy()
    master_df = master_df[master_df['MRNCmpdId'] != 'SMS0016578']
    master_slt_filepath = '/data/mridle/data/silent_live_test/live_files/all/' \
                          'out_features_data/features_master_slt_features.csv'
    if os.path.exists(master_slt_filepath):
        master_slt = get_slt_with_outcome()
    else:
        master_slt = pd.DataFrame()
    historic_data = pd.concat([master_df, master_slt], axis=0)

    historic_data['MRNCmpdId'] = historic_data['MRNCmpdId'].astype(str)
    features_df['MRNCmpdId'] = features_df['MRNCmpdId'].astype(str)

    hist_no_dupe = historic_data.merge(features_df[['FillerOrderNo', 'start_time']], how='left', indicator=True)
    hist_no_dupe = hist_no_dupe[hist_no_dupe['_merge'] == 'left_only']

    features_df = feature_no_show_before(features_df, hist_no_dupe)
    features_df['filename'] = filename

    model_dirs = Path(model_dir).glob('*')
    for model_dir in model_dirs:
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

    new_appts = features_df.merge(historic_data[['FillerOrderNo', 'start_time']], how='left', indicator=True)
    new_appts = new_appts[new_appts['_merge'] == 'left_only']
    new_appts.drop(columns=['_merge'], inplace=True)

    master_slt_updated = pd.concat([master_slt, new_appts], axis=0)
    master_slt_updated.drop_duplicates(inplace=True)
    master_slt_updated.to_csv(master_slt_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_path', type=str, help='Path to the input data')
    parser.add_argument('model_dir', type=str, help='Parent directory containing model subdirectories.')
    parser.add_argument('output_path', type=str, help='Destination to save the prediction data')
    parser.add_argument('start_date', type=str, help='Destination to save the prediction data')
    parser.add_argument('end_date', type=str, help='Destination to save the prediction data')
    parser.add_argument('file_encoding', type=str, help='Encoding arg for read_csv()')
    args = parser.parse_args()
    valid_date_range = (args.start_date, args.end_date)
    main(args.data_path, args.model_dir, args.output_path, valid_date_range)
