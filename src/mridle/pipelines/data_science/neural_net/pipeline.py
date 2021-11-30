from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_experiment,
                inputs=["train_data", "params:models.neural_net"],
                outputs=["neural_net_model", "neural_net_model_results"],
                name="train_neural_net_model"
            )
        ]
    )
