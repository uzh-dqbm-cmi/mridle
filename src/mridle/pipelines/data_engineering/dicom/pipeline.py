from kedro.pipeline import Pipeline, node
from .nodes import format_dicom_times_df, integrate_dicom_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=format_dicom_times_df,
                inputs="dicom_3_months_Christian",
                outputs="dicom_times_df",
                name="format_dicom_times_df"
            ),
            node(
                func=integrate_dicom_data,
                inputs=["slot_df", "dicom_3_months_Christian"],
                outputs="slot_w_dicom_df",
                name="integrate_dicom_data"
            ),
        ]
    )
