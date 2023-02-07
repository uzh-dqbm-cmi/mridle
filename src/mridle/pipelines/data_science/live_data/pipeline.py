from kedro.pipeline import Pipeline, node
from .nodes import get_slt_with_outcome, concat_master_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=get_slt_with_outcome,
                inputs=[],
                outputs="live_data",
                name="get_slt_with_outcome",
            ),
            node(
                func=concat_master_data,
                inputs=["master_feature_set_na_removed", 'live_data'],
                outputs="train_data_with_live",
                name="concat_master_data",
            )
        ]
    )
