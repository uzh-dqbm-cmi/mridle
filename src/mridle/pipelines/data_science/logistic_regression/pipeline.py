from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment
from .nodes import remove_na


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=remove_na,
                inputs=["master_feature_set"],
                outputs=["input_feature_set"],
                name="remove_na"
            ),
            node(
                func=run_experiment,
                inputs=["input_feature_set", "params:models.random_forest"],
                outputs=["random_forest_model", "random_forest_model_results"],
                name="train_random_forest_model"
            )
        ]
    )
