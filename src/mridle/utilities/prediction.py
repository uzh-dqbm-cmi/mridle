import argparse
import pandas as pd
from pathlib import Path
import pickle
from mridle.pipelines.data_engineering.ris.nodes import build_status_df
from mridle.pipelines.data_science.feature_engineering.nodes import build_feature_set, remove_na
from mridle.experiment.experiment import Experiment
from mridle.experiment.data_set import DataSet


def main(data_path, model_dir, output_path):
    """
    Make predictions for all models in model_dir on the given data, saving the resulting predictions to output_path.
    Args:
        data_path: Path to input data (csv).
        model_dir: Directory to the model experiments. This directory should contain a set of directories containing
         pickled serialized `Experiments`.
        output_path: Destination to save the predictions file to (csv).

    Returns: None
    """
    raw_df = pd.read_csv(data_path)
    status_df = build_status_df(raw_df)
    features_df_maybe_na = build_feature_set(status_df)
    features_df = remove_na(features_df_maybe_na)
    prediction_df = features_df.copy()

    model_dirs = Path(model_dir).glob('*')
    for model_dir in model_dirs:
        model_paths = model_dir.glob('*')
        for model_path in model_paths:
            serialized_model = pickle.load(model_path)
            exp = Experiment.deserialize(serialized_model)
            data_set = DataSet(exp.dataset.config, features_df)
            preds_proba = exp.final_predictor.predict_proba(data_set.x)
            model_name = exp.metadata.get('name', model_path.name)
            prediction_df[f'prediction_{model_name}'] = preds_proba

    prediction_df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('data_path',  type=str, help='Path to the input data')
    parser.add_argument('model_dir',  type=str, help='Parent directory containing model subdirectories.')
    parser.add_argument('output_path',  type=str, help='Destination to save the prediction data')
    args = parser.parse_args()
    main(args.data_path, args.model_dir, args.output_path)
