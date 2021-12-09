from kedro.pipeline import Pipeline, node
from .nodes import build_feature_set, train_val_split, remove_na


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_feature_set,
                inputs=["status_df", "params:ris.valid_date_range"],
                outputs="master_feature_set",
                name="build_feature_set",
            ),
            node(
                func=remove_na,
                inputs="master_feature_set",
                outputs="master_feature_set_na_removed",
                name="remove_na"
            ),
            node(
                func=train_val_split,
                inputs=["master_feature_set_na_removed", "params:train_val_split"],
                outputs=["train_data", "validation_data"],
                name="train_val_split"
            )
        ]
    )
