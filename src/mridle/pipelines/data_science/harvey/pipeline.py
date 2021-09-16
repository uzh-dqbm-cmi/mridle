from kedro.pipeline import Pipeline, node
from .nodes import build_harvey_et_al_features_set, process_features_for_model, train_harvey_model_logistic_reg,\
    train_harvey_model_random_forest, plot_harvey_metrics


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_harvey_et_al_features_set,
                inputs="status_df",
                outputs="harvey_feature_set",
                name="build_harvey_et_al_features_set",
            ),
            node(
                func=process_features_for_model,
                inputs=["harvey_feature_set", "parameters"],
                outputs="harvey_model_input",
                name="process_features_for_harvey_model"
            ),
            node(
                func=train_harvey_model_logistic_reg,
                inputs=["harvey_model_input", "params:models.harvey_logistic_reg"],
                outputs=["harvey_model_logistic_reg", "harvey_model_results_logistic_reg"],
                name="train_harvey_model_logistic_reg"
            ),
            node(
                func=train_harvey_model_random_forest,
                inputs=["harvey_model_input", "params:models.harvey_random_forest"],
                outputs=["harvey_model_random_forest", "harvey_model_results_random_forest"],
                name="train_harvey_model_random_forest"
            ),
            node(
                func=lambda lr, rf: {'Logistic Regression': lr, 'Random Forest': rf, },
                inputs=['harvey_model_results_logistic_reg', 'harvey_model_results_random_forest'],
                outputs='harvey_models_results_dict',
                name='build_harvey_models_results_dict'
            ),
            node(
                func=plot_harvey_metrics,
                inputs='harvey_models_results_dict',
                outputs="harvey_model_metrics_plot",
                name='plot_harvey_metrics'
            ),
        ]
    )
