from kedro.pipeline import Pipeline, node
from .nodes import create_evaluation_table, create_model_precision_comparison_plot, plot_pr_roc_curve_comparison, \
    plot_permutation_importance_charts


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_evaluation_table,
                inputs=["harvey_model_logistic_reg", "harvey_model_random_forest", "logistic_regression_model",
                        "random_forest_model", "xgboost_model", "neural_net_model", "validation_data"],
                outputs="evaluation_table",
                name="create_evaluation_table"
            ),
            node(
                func=create_model_precision_comparison_plot,
                inputs="evaluation_table",
                outputs="model_precision_comparison_plot",
                name="create_model_precision_comparison_plot"
            ),
            node(
                func=plot_pr_roc_curve_comparison,
                inputs=["harvey_model_logistic_reg", "harvey_model_random_forest", "logistic_regression_model",
                        "random_forest_model", "xgboost_model", "neural_net_model", "validation_data"],
                outputs=["pr_curve_comparison", "roc_curve_comparison"],
                name="plot_pr_roc_curve_comparison"
            ),
            node(
                func=plot_permutation_importance_charts,
                inputs=["harvey_model_logistic_reg", "harvey_model_random_forest", "logistic_regression_model",
                        "random_forest_model", "xgboost_model", "neural_net_model", "train_data", "validation_data"],
                outputs=["harvey_logistic_reg_permutation_imp_train", "harvey_logistic_reg_permutation_imp_validation",
                         "harvey_random_forest_permutation_imp_train",
                         "harvey_random_forest_permutation_imp_validation",
                         "logistic_regression_permutation_imp_train", "logistic_regression_permutation_imp_validation",
                         "random_forest_permutation_imp_train", "random_forest_permutation_imp_validation",
                         "xgboost_permutation_imp_train", "xgboost_permutation_imp_validation",
                         "neural_net_permutation_imp_train", "neural_net_permutation_imp_validation"
                         ],
                name="plot_permutation_imp_xgboost"
            )
        ]
    )
