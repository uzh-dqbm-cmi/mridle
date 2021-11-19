import pandas as pd
import altair as alt
from mridle.experiment.dataset import DataSet
from mridle.experiment.experiment import Experiment
from sklearn.metrics import f1_score, confusion_matrix, brier_score_loss
import numpy as np


def create_evaluation_table(harvey_model_log_reg, harvey_random_forest, logistic_regression_model, random_forest_model,
                            xgboost_model, validation_data):

    serialised_models = [('Harvey LogReg', harvey_model_log_reg), ('Harvey RandomForest', harvey_random_forest),
                         ('Logistic Regression', logistic_regression_model), ('RandomForest', random_forest_model),
                         ('XGBoost', xgboost_model)]

    evaluation_table = []
    avg_appts_per_week = 166  # taken from aggregation of df_features_original data for the year 2017 (in notebook 52)

    for (model_name, serialised_m) in serialised_models:

        val_dataset = DataSet(serialised_m['components']['DataSet']['config'], validation_data)

        experiment = Experiment.deserialize(serialised_m)

        preds_prob = experiment.model.predict_proba(val_dataset.x)

        preds_prob_sorted = np.sort(preds_prob)[::-1]

        calc_list = []

        for i in range(1, 11):
            top_i_cut = preds_prob_sorted[int(np.round(0.06 * i * len(preds_prob_sorted)))]
            preds_i_pw = (top_i_cut, np.where(preds_prob > top_i_cut, 1, 0))
            calc_list.append(preds_i_pw)

        for (c, p) in calc_list:
            F1 = f1_score(val_dataset.y, p, average='macro')
            brier_loss = brier_score_loss(val_dataset.y, preds_prob)
            TN, FP, FN, TP = confusion_matrix(val_dataset.y, p).ravel()
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            # Precision or positive predictive value
            PPV = TP / (TP + FP)
            # Negative predictive value
            NPV = TN / (TN + FN)
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            # False negative rate
            FNR = FN / (TP + FN)
            # False discovery rate
            FDR = FP / (TP + FP)
            # Overall accuracy
            ACC = (TP + TN) / (TP + FP + FN + TN)
            # num no-shows predicted
            num_no_shows_pred = np.sum(p)
            # num no-shows actual
            num_no_shows_act = np.sum(val_dataset.y)
            total_num_appts = len(val_dataset.y)
            percentage_predicted = (num_no_shows_pred / total_num_appts)

            predictions_per_week = avg_appts_per_week * percentage_predicted
            correct_per_week = predictions_per_week * PPV

            metrics = [model_name, c, F1, brier_loss, TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, num_no_shows_pred,
                       num_no_shows_act, total_num_appts,
                       percentage_predicted, predictions_per_week, correct_per_week]

            evaluation_table.append(metrics)

    evaluation_table = pd.DataFrame(evaluation_table,
                                    columns=['Model', 'Cut-off', 'F1', 'brier_loss', 'TPR / Recall / Sensitivity',
                                             'TNR / Specificity', 'PPV / Precision', 'NPV',
                                             'FPR', 'FNR', 'FDR', 'ACC', '# Predicted No Shows', '# Actual No Shows',
                                             'Total number of appts', 'Percentage predicted to no-show',
                                             '# No-show predictions per week', 'Of which were no-shows'])
    evaluation_table.iloc[:, :15] = np.round(evaluation_table.iloc[:, :15], 3)
    evaluation_table.iloc[:, 15:] = np.round(evaluation_table.iloc[:, 15:], 1)
    return evaluation_table


def create_model_precision_comparison_plot(evaluation_table_df: pd.DataFrame) -> alt.Chart:
    """
    Taking in a dataframe containing evaluation results achieved by each model, produce a scatter plot with the
    number of positive predictions on the X-axis, and the precision on the Y-axis

    Args:
        evaluation_table_df: dataframe of model evaluations on a test set returned by the create_evaluation_table
        function/node

    Returns:
        altair scatter plot with number of positive predictions on the X-axis, and the precision on the Y-axis

    """
    model_precision_comparison_plot = alt.Chart(evaluation_table_df[['# No-show predictions per week',
                                                                     'PPV / Precision',
                                                                     'model_name'
                                                                     ]]
                                                ).mark_circle(size=60, opacity=1).encode(
        x='# No-show predictions per week',
        y=alt.Y('PPV / Precision', scale=alt.Scale(domain=(0, 1))),
        color='model_name'
    ).properties(width=500)

    return model_precision_comparison_plot
