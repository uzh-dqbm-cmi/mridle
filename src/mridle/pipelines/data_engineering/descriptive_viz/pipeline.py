from kedro.pipeline import Pipeline, node
from .nodes import plot_dave_b


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=plot_dave_b,
                inputs=["slot_df", "slot_w_dicom_df"],
                outputs="dave_b",
                name="plot_dave_b",
            ),

        ]
    )
