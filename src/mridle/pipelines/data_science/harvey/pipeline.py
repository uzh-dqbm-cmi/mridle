from kedro.pipeline import Pipeline, node
from .nodes import build_harvey_et_al_features_set, process_features_for_model, train_harvey_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_harvey_et_al_features_set,
                inputs="status_df",
                outputs="harvey_feature_set",
                name="build_harvey_et_al_features_set",
            ),
            node(
                func=process_features_for_model,
                inputs=["harvey_feature_set", "parameters"],
                outputs="harvey_model_input",
                name="process_features_for_harvey_model"
            ),
            node(
                func=train_harvey_model,
                inputs=["harvey_model_input", "params:models.harvey"],
                outputs=["harvey_model", "harvey_model_results"],
                name="train_harvey_model"
            ),
        ]
    )
