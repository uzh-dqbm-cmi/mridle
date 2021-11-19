from kedro.pipeline import Pipeline, node
from .nodes import build_feature_set, train_val_split


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_feature_set,
                inputs="status_df",
                outputs="master_feature_set",
                name="build_feature_set",
            ),
            node(
                func=train_val_split,
                inputs="master_feature_set",
                outputs=["train_data", "validation_data"],
                name="train_val_split"
            )
        ]
    )
