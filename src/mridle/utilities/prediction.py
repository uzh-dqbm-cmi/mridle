import argparse
import pandas as pd
from pathlib import Path
import pickle

from mridle.pipelines.data_engineering.ris.nodes import build_status_df, prep_raw_df_for_parquet
from mridle.pipelines.data_science.feature_engineering.nodes import build_feature_set, remove_na
from mridle.experiment.experiment import Experiment
from mridle.experiment.dataset import DataSet


def main(data_path, model_dir, output_path, valid_date_range, file_encoding, master_feature_set, rfs_df):
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
    if file_encoding:
        raw_df = pd.read_csv(data_path, encoding=file_encoding)
    else:
        raw_df = pd.read_csv(data_path)

    exclude_pat_ids = list()  # TODO!
    print(raw_df[raw_df['MRNCmpdId'].str.contains('_')].head())
    raw_df['MRNCmpdId'] = raw_df['MRNCmpdId'].str.replace('_', '')  # because some MRNCmpdIds have leading underscores,
    # which usually denotes a test appointment and is thus removed, but in silent live test and with 'future' data,
    # sometimes the MRNCmpdId has a leading underscore which is later removed/changed to a 'proper' MRNCmpdId
    print(raw_df[raw_df['MRNCmpdId'] == '2570441'])  # because some MRNCmpdIds have leading underscores,

    formatted_df = prep_raw_df_for_parquet(raw_df)
    status_df = build_status_df(formatted_df, exclude_pat_ids)
    status_df = status_df.merge(rfs_df, how='left')

    features_df_maybe_na = build_feature_set(status_df, valid_date_range, build_future_slots=True)
    features_df = remove_na(features_df_maybe_na)

    # Get number of previous no shows from historical data and add to data set
    master_df = master_feature_set.copy()
    prev_no_shows = master_df[['MRNCmpdId', 'no_show_before']].groupby('MRNCmpdId').max().reset_index()
    prev_no_shows['MRNCmpdId'] = prev_no_shows['MRNCmpdId'] .astype(int)
    features_df['MRNCmpdId'] = features_df['MRNCmpdId'] .astype(int)

    features_df = features_df.merge(prev_no_shows, on=['MRNCmpdId'], how='left', suffixes=['_current', '_hist'])
    features_df['no_show_before_hist'].fillna(0, inplace=True)
    features_df['no_show_before'] = features_df['no_show_before_current'] + features_df['no_show_before_hist']
    features_df.drop(['no_show_before_current', 'no_show_before_hist'], axis=1, inplace=True)
    features_df['no_show_before_sq'] = features_df['no_show_before'] ** 2

    prediction_df = features_df.copy()

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
            prediction_df[f'prediction_{model_name}'] = preds_proba

    prediction_df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_path',  type=str, help='Path to the input data')
    parser.add_argument('model_dir',  type=str, help='Parent directory containing model subdirectories.')
    parser.add_argument('output_path',  type=str, help='Destination to save the prediction data')
    parser.add_argument('start_date',  type=str, help='Destination to save the prediction data')
    parser.add_argument('end_date',  type=str, help='Destination to save the prediction data')
    parser.add_argument('file_encoding',  type=str, help='Encoding arg for read_csv()')
    args = parser.parse_args()
    valid_date_range = (args.start_date, args.end_date)
    main(args.data_path, args.model_dir, args.output_path, valid_date_range)
