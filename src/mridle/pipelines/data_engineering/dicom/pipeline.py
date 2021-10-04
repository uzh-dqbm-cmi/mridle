from kedro.pipeline import Pipeline, node
from .nodes import preprocess_dicom_data, aggregate_dicom_images, integrate_dicom_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_dicom_data,
                inputs=["dicom_5_years_rdsc", "id_list_df"],
                outputs="dicom_series_metadata",
                name="preprocess_dicom_data"
            ),
            node(
                func=aggregate_dicom_images,
                inputs="dicom_series_metadata",
                outputs="dicom_times_df",
                name="aggregate_dicom_metadata"
            ),
            node(
                func=integrate_dicom_data,
                inputs=["slot_df", "dicom_times_df"],
                outputs="slot_w_dicom_df",
                name="integrate_dicom_data"
            ),
        ]
    )
