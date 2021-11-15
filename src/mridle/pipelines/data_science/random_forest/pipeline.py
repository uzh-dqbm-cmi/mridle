from kedro.pipeline import Pipeline, node
from mridle.utilities.modeling import run_experiment


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=run_experiment,
                inputs=["random_forest_model_input", "params:models.random_forest"],
                outputs=["harvey_model_random_forest", "harvey_model_results_random_forest"],
                name="train_harvey_model_random_forest"
            ),
            node(
                func=lambda lr, rf: {'Logistic Regression': lr, 'Random Forest': rf, },
                inputs=['harvey_model_results_logistic_reg', 'harvey_model_results_random_forest'],
                outputs='harvey_models_results_dict',
                name='build_harvey_models_results_dict'
            ),
        ]
    )
