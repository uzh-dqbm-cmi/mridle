from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_experiment,
                inputs=["master_feature_set_na_removed", "params:models.logistic_regression"],
                outputs=["logistic_regression_model", "logistic_regression_model_results"],
                name="train_logistic_regression_model"
            )
        ]
    )
