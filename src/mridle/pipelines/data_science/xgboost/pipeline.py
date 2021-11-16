from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment
from .nodes import process_features_for_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=process_features_for_model,
                inputs=["master_feature_set"],
                outputs="model_input",
                name="process_features_for_xgboost_model"
            ),

            node(
                func=run_experiment,
                inputs=["model_input", "params:models.xgboost"],
                outputs=["xgboost_model", "xgboost_model_results"],
                name="train_xgboost_model"
            )
        ]
    )
