from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_experiment,
                inputs=["master_feature_set", "params:models.xgboost"],
                outputs=["xgboost_model", "xgboost_model_results"],
                name="train_xgboost_model"
            )
        ]
    )
