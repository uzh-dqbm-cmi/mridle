from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment
from .nodes import process_features_for_model, plot_harvey_metrics


def create_pipeline(**kwargs):
    return Pipeline(
        [

            node(
                func=process_features_for_model,
                inputs=["master_feature_set"],
                outputs="harvey_model_input",
                name="process_features_for_harvey_model"
            ),
            node(
                func=run_experiment,
                inputs=["harvey_model_input", "params:models.harvey_logistic_reg"],
                outputs=["harvey_model_logistic_reg", "harvey_model_results_logistic_reg"],
                name="train_harvey_model_logistic_reg"
            ),
            node(
                func=run_experiment,
                inputs=["harvey_model_input", "params:models.harvey_random_forest"],
                outputs=["harvey_model_random_forest", "harvey_model_results_random_forest"],
                name="train_harvey_model_random_forest"
            ),
            node(
                func=run_experiment,
                inputs=["harvey_model_input", "params:models.harvey_random_forest_hp"],
                outputs=["harvey_model_random_forest_hp", "harvey_model_results_random_forest_hp"],
                name="train_harvey_model_random_forest_hp"
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
