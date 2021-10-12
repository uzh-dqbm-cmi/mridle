from kedro.pipeline import Pipeline, node
from .nodes import build_dispo_exp_1_df, build_dispo_exp_2_df, calc_exp_confusion_matrix, calc_jaccard_score_table
from ..ris.nodes import build_status_df, build_slot_df


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # EXPERIMENT 1 - DEPRECATED
            node(
                func=lambda part_1, part_2: part_1 + part_2,
                inputs=["experiment1A", "experiment1B"],
                outputs="experiment1_records",
                name="concat_raw_exp_1_records",
            ),
            node(
                func=build_dispo_exp_1_df,
                inputs=["experiment1_records", "test_patient_ids"],
                outputs="dispo_exp_1_df",
                name="build_dispo_exp_1_df",
            ),
            node(
                func=calc_exp_confusion_matrix,
                inputs=["dispo_exp_1_df", "slot_df"],
                outputs="val_exp_1_confusion_matrix",
                name="calc_exp_1_confusion_matrix"

            ),

            # EXPERIMENT 2 - DEVELOPMENT
            node(
                func=lambda part_1, part_2: part_1 + part_2,
                inputs=["experiment2", "experiment2_corrections"],
                outputs="experiment_2_records",
                name="concat_raw_exp_2_records"
            ),
            node(
                func=build_dispo_exp_2_df,
                inputs=["experiment_2_records", "test_patient_ids"],
                outputs="dispo_development_slot_df",
                name="build_dispo_exp_2_df"
            ),
            node(
                func=build_status_df,
                inputs=["val_ris_development", "test_patient_ids"],
                outputs="val_ris_development_status_df",
                name="build_val_ris_development_status_df"
            ),
            node(
                func=build_slot_df,
                inputs="val_ris_development_status_df",
                outputs="val_ris_development_slot_df",
                name="build_val_ris_development_slot_df"
            ),

            # EXPERIMENT 3 - EVALUATION
            node(
                func=lambda part_1, part_2: part_1 + part_2,
                inputs=["experiment3", "experiment3_corrections"],
                outputs="experiment_3_records",
                name="concat_raw_exp_3_records"
            ),
            node(
                func=build_dispo_exp_2_df,
                inputs=["experiment_3_records", "test_patient_ids"],
                outputs="dispo_evaluation_slot_df",
                name="build_dispo_exp_3_df"
            ),
            node(
                func=build_status_df,
                inputs=["val_ris_evaluation", "test_patient_ids"],
                outputs="val_ris_evaluation_status_df",
                name="build_val_ris_evaluation_status_df"
            ),
            node(
                func=build_slot_df,
                inputs="val_ris_evaluation_status_df",
                outputs="val_ris_evaluation_slot_df",
                name="build_val_ris_evaluation_slot_df"
            ),

            # REPORTING
            node(
                func=calc_jaccard_score_table,
                inputs=["dispo_development_slot_df", "val_ris_development_slot_df", "dispo_evaluation_slot_df",
                        "val_ris_evaluation_slot_df"],
                outputs="validation_jaccard_table",
                name="calc_jaccard_score_table"
            ),
            node(
                func=calc_exp_confusion_matrix,
                inputs=["dispo_development_slot_df", "val_ris_development_slot_df"],
                outputs="val_development_confusion_matrix",
                name="calc_development_exp_confusion_matrix"
            ),
            node(
                func=calc_exp_confusion_matrix,
                inputs=["dispo_evaluation_slot_df", "val_ris_evaluation_slot_df"],
                outputs="val_evaluation_confusion_matrix",
                name="calc_evaluation_exp_confusion_matrix"
            )
        ]
    )
