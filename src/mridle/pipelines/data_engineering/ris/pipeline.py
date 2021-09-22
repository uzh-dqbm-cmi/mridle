from kedro.pipeline import Pipeline, node
import pandas as pd
from .nodes import prep_raw_df_for_parquet, build_status_df, build_slot_df


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=lambda *dfs: pd.concat(dfs),
                inputs=["ris_2015_2016", "ris_2017_2018", "ris_2019"],
                outputs="concatenated_raw_ris_five_years",
                name="concat_raw_data",
            ),
            node(
                func=prep_raw_df_for_parquet,
                inputs="concatenated_raw_ris_five_years",
                outputs="raw_ris_five_years",
                name="prep_raw_df_for_parquet",
            ),
            node(
                func=build_status_df,
                inputs=["raw_ris_five_years", "test_patient_ids"],
                outputs="status_df",
                name="build_status_df"
            ),
            node(
                func=build_slot_df,
                inputs="status_df",
                outputs="slot_df",
                name="build_slot_df"
            ),
        ]
    )
