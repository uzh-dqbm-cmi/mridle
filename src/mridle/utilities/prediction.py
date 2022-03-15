import argparse
import pandas as pd
from pathlib import Path
import pickle
from mridle.pipelines.data_engineering.ris.nodes import build_status_df, prep_raw_df_for_parquet
from mridle.pipelines.data_science.feature_engineering.nodes import build_feature_set, remove_na
from mridle.experiment.experiment import Experiment
from mridle.exblox.dataset import DataSet


def main(data_path, model_dir, output_path, valid_date_range, file_encoding):
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
    formatted_df = prep_raw_df_for_parquet(raw_df)
    status_df = build_status_df(formatted_df, exclude_pat_ids)
    features_df_maybe_na = build_feature_set(status_df, valid_date_range, build_future_slots=True)
    features_df = remove_na(features_df_maybe_na)
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
