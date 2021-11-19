from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_experiment,
                inputs=["master_feature_set_na_removed", "params:models.random_forest"],
                outputs=["random_forest_model", "random_forest_model_results"],
                name="train_random_forest_model"
            )
        ]
    )
