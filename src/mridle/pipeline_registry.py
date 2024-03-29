# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from mridle.pipelines.data_engineering import ris, dicom, dispo
from mridle.pipelines.data_science import harvey, feature_engineering, descriptive_viz, random_forest, xgboost, \
    logistic_regression, neural_net, model_comparison, live_data, xgboost_with_live


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    ris_pipeline = ris.create_pipeline()
    dicom_pipeline = dicom.create_pipeline()
    dispo_pipeline = dispo.create_pipeline()
    descriptive_viz_pipeline = descriptive_viz.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    live_data_pipeline = live_data.create_pipeline()
    harvey_pipeline = harvey.create_pipeline()
    logistic_regression_pipeline = logistic_regression.create_pipeline()
    random_forest_pipeline = random_forest.create_pipeline()
    xgboost_pipeline = xgboost.create_pipeline()
    xgboost_with_live_pipeline = xgboost_with_live.create_pipeline()
    neural_net_pipeline = neural_net.create_pipeline()
    model_comparison_pipeline = model_comparison.create_pipeline()

    return {

        "__default__": ris_pipeline + feature_engineering_pipeline + harvey_pipeline + logistic_regression_pipeline +
        random_forest_pipeline + xgboost_pipeline + model_comparison_pipeline,
        "all": ris_pipeline + dicom_pipeline + dispo_pipeline + descriptive_viz_pipeline +
        feature_engineering_pipeline + harvey_pipeline + logistic_regression_pipeline +
        random_forest_pipeline + xgboost_pipeline + neural_net_pipeline + model_comparison_pipeline,
        "ris": ris_pipeline,
        "dicom": dicom_pipeline,
        "dispo": dispo_pipeline,
        "descriptive_viz": descriptive_viz_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "live_data": live_data_pipeline,
        "harvey": harvey_pipeline,
        "logistic_regression": logistic_regression_pipeline,
        "random_forest": random_forest_pipeline,
        "xgboost": xgboost_pipeline,
        "xgboost_with_live": xgboost_with_live_pipeline,
        "neural_net": neural_net_pipeline,
        "model_comparison": model_comparison_pipeline,
        "models": harvey_pipeline + logistic_regression_pipeline + random_forest_pipeline + xgboost_pipeline
    }
