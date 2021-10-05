from kedro.pipeline import Pipeline, node
from .nodes import preprocess_dicom_data, aggregate_dicom_images, integrate_dicom_data, generate_idle_time_stats, \
    prep_terminplanner


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
            node(
                func=prep_terminplanner,
                inputs="terminplanner_df",
                outputs="terminplanner_aggregated_df",
                name="prep_terminplanner"
            ),
            node(
                func=generate_idle_time_stats,
                inputs=["dicom_times_df", "terminplanner_aggregated_df"],
                outputs=["appts_and_gaps", "daily_idle_stats"],
                name="generate_idle_time_stats"
            )
        ]
    )
