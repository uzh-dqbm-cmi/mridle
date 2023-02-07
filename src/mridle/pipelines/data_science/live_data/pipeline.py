from kedro.pipeline import Pipeline, node
from .nodes import get_slt_with_outcome


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_slt_with_outcome,
                inputs=[""],
                outputs="live_data",
                name="get_slt_with_outcome",
            )
        ]
    )
