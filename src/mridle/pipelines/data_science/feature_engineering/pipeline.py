from kedro.pipeline import Pipeline, node
from .nodes import build_feature_set


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_feature_set,
                inputs="status_df",
                outputs="master_feature_set",
                name="build_feature_set",
            )
        ]
    )
