from kedro.pipeline import Pipeline, node
from .nodes import plot_dave_b, plot_no_show_by_day_of_week, plot_no_show_by_month, plot_no_show_by_hour_of_day,\
    plot_no_show_by_age, plot_appts_per_patient, plot_no_show_heat_map, plot_appt_noshow_tree_map, \
    plot_numerical_feature_correlations
from functools import partial


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=plot_dave_b,
                inputs=["slot_df", "slot_w_dicom_df", "params:ris.dataset_valid_starting",
                        "params:ris.dataset_valid_ending"],
                outputs="dave_b",
                name="plot_dave_b",
            ),
            node(
                func=plot_no_show_by_day_of_week,
                inputs="master_feature_set",
                outputs="no_show_by_day_of_week",
                name="plot_no_show_by_day_of_week",
            ),
            node(
                func=plot_no_show_by_month,
                inputs="master_feature_set",
                outputs="no_show_by_month",
                name="plot_no_show_by_month",
            ),
            node(
                func=plot_no_show_by_hour_of_day,
                inputs="master_feature_set",
                outputs="no_show_by_hour_of_day",
                name="plot_no_show_by_hour_of_day",
            ),
            node(
                func=plot_no_show_by_age,
                inputs="master_feature_set",
                outputs="no_show_by_age",
                name="plot_no_show_by_age",
            ),
            node(
                func=plot_appts_per_patient,
                inputs="master_feature_set",
                outputs="appts_per_patient",
                name="plot_appts_per_patient",
            ),
            node(
                func=plot_no_show_heat_map,
                inputs="master_feature_set",
                outputs="no_show_heat_map",
                name="plot_no_show_heat_map",
            ),
            node(
                func=partial(plot_no_show_heat_map, log=True),
                inputs="master_feature_set",
                outputs="no_show_heat_map_log",
                name="plot_no_show_heat_map_log",
            ),
            node(
                func=plot_appt_noshow_tree_map,
                inputs="master_feature_set",
                outputs="appt_noshow_tree_map",
                name="plot_appt_noshow_tree_map",
            ),
            node(
                func=plot_numerical_feature_correlations,
                inputs="master_feature_set",
                outputs=["numerical_feature_correlations", "correlation_list"],
                name="plot_numerical_feature_correlations",
            )
        ]
    )
