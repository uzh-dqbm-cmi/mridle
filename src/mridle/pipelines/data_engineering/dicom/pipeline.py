from kedro.pipeline import Pipeline, node
from .nodes import preprocess_dicom_data, aggregate_dicom_images, integrate_dicom_data, generate_idle_time_stats, \
    aggregate_terminplanner, generate_idle_time_plots, subset_valid_appts, fill_in_terminplanner_gaps


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=subset_valid_appts,
                inputs=["dicom_5_years_rdsc", "valid_dicom_ids_2016_2019"],
                outputs="dicom_data_valid_appts",
                name="subset_valid_appts"
            ),
            node(
                func=preprocess_dicom_data,
                inputs="dicom_data_valid_appts",
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
                func=aggregate_terminplanner,
                inputs="raw_terminplanner_df",
                outputs="terminplanner_aggregated_df",
                name="aggregate_terminplanner"
            ),
            node(
                func=fill_in_terminplanner_gaps,
                inputs="terminplanner_aggregated_df",
                outputs="terminplanner_df",
                name="aggregate_terminplanner"
            ),
            node(
                func=generate_idle_time_stats,
                inputs=["dicom_times_df", "terminplanner_df"],
                outputs=["appts_and_gaps", "daily_idle_stats"],
                name="generate_idle_time_stats"
            ),
            node(
                func=generate_idle_time_plots,
                inputs=["appts_and_gaps", "daily_idle_stats"],
                outputs=["daily_idle_buffer_active_percentages_plot", "full_zebra", "one_week_zebra"],
                name="generate_plots"
            )
        ]
    )
