from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment
from .nodes import plot_harvey_metrics


def create_pipeline(**kwargs):
    return Pipeline(
        [

            node(
                func=run_experiment,
                inputs=["train_data", "params:models.harvey_logistic_reg"],
                outputs=["harvey_model_logistic_reg", "harvey_model_results_logistic_reg"],
                name="train_harvey_model_logistic_reg"
            ),
            node(
                func=run_experiment,
                inputs=["train_data", "params:models.harvey_random_forest"],
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
