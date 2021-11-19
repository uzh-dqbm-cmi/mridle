from kedro.pipeline import Pipeline, node
from .nodes import create_evaluation_table, create_model_precision_comparison_plot


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_evaluation_table,
                inputs=["logistic_regression_model",
                        "random_forest_model", "xgboost_model", "validation_data"],
                outputs="evaluation_table",
                name="create_evaluation_table"
            ),
            node(
                func=create_model_precision_comparison_plot,
                inputs="evaluation_table",
                outputs="model_precision_comparison_plot",
                name="create_model_precision_comparison_plot"
            )
        ]
    )
