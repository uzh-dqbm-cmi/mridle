import pandas as pd
import altair as alt
from mridle.experiment.dataset import DataSet
from mridle.experiment.experiment import Experiment
from sklearn.metrics import f1_score, confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve, auc,\
    make_scorer, log_loss
import numpy as np
# from sklearn.inspection import permutation_importance


def create_evaluation_table(harvey_model_log_reg, harvey_random_forest, logistic_regression_model, random_forest_model,
                            xgboost_model, neural_net_model, validation_data):
    """
    Function to create a table of metrics for the models.

    For each model, we create predictions on the validation data and order these from largest to smallest. We then take
    the top 10 of these predictions, mark them as a no show prediction (the rest of the appointments marked as not to
    no-show), and calculate a set of statistics based on this (e.g. F1 macro score). This is replicating the accuracy
    metrics we can expect to have if we provide the 10 appointments most likely to no-show (according to the model) to
    the nurses. We then repeat this step for the top 20, 30, 40, ..., 100. This creates a large table of 10 rows per
    model (1 row per top X) containing accuracy metrics.

    Args:
        harvey_model_log_reg: serialised harvey logistic regression model
        harvey_random_forest: serialised harvey random forest model
        logistic_regression_model: serialised logistic regression model
        random_forest_model: serialised random forest model
        xgboost_model: serialised xgboost model
        neural_net_model: serialised neural net model
        validation_data: validation data, split out from master_feature_set before experiments were ran

    Returns:

    """
    serialised_models = [('Harvey LogReg', harvey_model_log_reg), ('Harvey RandomForest', harvey_random_forest),
                         ('Logistic Regression', logistic_regression_model), ('RandomForest', random_forest_model),
                         ('XGBoost', xgboost_model), ('Neural Net', neural_net_model)]

    evaluation_table = []
    avg_appts_per_week = 166  # taken from aggregation of df_features_original data for the year 2017 (in notebook 52)

    for (model_name, serialised_m) in serialised_models:
        model_validation_data = validation_data.copy()

        val_dataset = DataSet(serialised_m['components']['DataSet']['config'], model_validation_data)

        experiment = Experiment.deserialize(serialised_m)

        preds_prob = experiment.final_predictor.predict_proba(val_dataset.x)
        if model_name == 'Neural Net':
            preds_prob = [prob for [prob] in preds_prob]
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
                                                                     'Model'
                                                                     ]]
                                                ).mark_circle(size=60, opacity=1).encode(
        x='# No-show predictions per week',
        y=alt.Y('PPV / Precision', scale=alt.Scale(domain=(0, 1))),
        color='Model'
    ).properties(width=500)

    # Add horizontal line to chart, showing the precision which would be achieved through random guessing
    no_show_rate = evaluation_table_df.loc[0, '# Actual No Shows'] / evaluation_table_df.loc[0, 'Total number of appts']

    line = pd.DataFrame({
        '# No-show predictions per week': [0, 100],
        'PPV / Precision': [no_show_rate, no_show_rate],
        'Model': ['Baseline (Random Guessing)', 'Baseline (Random Guessing)']
    })

    line_plot = alt.Chart(line).mark_line(size=2).encode(
        x='# No-show predictions per week',
        y='PPV / Precision',
        color='Model'
    )

    return model_precision_comparison_plot + line_plot


def plot_pr_roc_curve_comparison(harvey_model_log_reg, harvey_random_forest, logistic_regression_model,
                                 random_forest_model, xgboost_model, neural_net_model, validation_data):

    serialised_models = [('Harvey LogReg', harvey_model_log_reg), ('Harvey RandomForest', harvey_random_forest),
                         ('Logistic Regression', logistic_regression_model), ('RandomForest', random_forest_model),
                         ('XGBoost', xgboost_model),  ('Neural Net', neural_net_model)]

    alt.data_transformers.disable_max_rows()
    all_pr_df = pd.DataFrame()
    all_roc_df = pd.DataFrame()

    for (model_name, serialised_m) in serialised_models:
        model_validation_data = validation_data.copy()

        val_dataset = DataSet(serialised_m['components']['DataSet']['config'], model_validation_data)

        experiment = Experiment.deserialize(serialised_m)

        p, r, t = precision_recall_curve(val_dataset.y, experiment.final_predictor.predict_proba(val_dataset.x))
        pr_df = pd.DataFrame()
        pr_df['p'] = p
        pr_df['r'] = r
        pr_df['name'] = '{}: {}'.format(model_name, round(auc(r, p), 3))

        all_pr_df = pd.concat([all_pr_df, pr_df], axis=0)

        fpr, tpr, thresholds = roc_curve(val_dataset.y, experiment.final_predictor.predict_proba(val_dataset.x))
        roc_df = pd.DataFrame()
        roc_df['fpr'] = fpr
        roc_df['tpr'] = tpr
        roc_df['name'] = '{}: {}'.format(model_name, round(auc(fpr, tpr), 3))

        all_roc_df = pd.concat([all_roc_df, roc_df], axis=0)

    all_pr_df_mean = all_pr_df.groupby(['r', 'name']).mean().reset_index()

    pr_curves = alt.Chart(all_pr_df_mean).mark_line(color='red').encode(
        alt.X('r', title="Recall"),
        alt.Y('p', title="Precision"),
        color=alt.Color('name', legend=alt.Legend(
            orient='top-right',
            title='',
            fillColor='white',
            titleAnchor='middle'))
    ).properties(title='Precision Recall Curves')

    baseline_df = pd.DataFrame({'y': [np.sum(val_dataset.y) / len(val_dataset.y)]})
    hline = alt.Chart(baseline_df).mark_rule(color='black', strokeWidth=2, strokeDash=[1, 1]).encode(y='y:Q')
    pr_curves = pr_curves + hline

    roc_curves = alt.Chart(all_roc_df).mark_line(color='red').encode(
        alt.X('fpr', title="False Positive Rate"),
        alt.Y('tpr', title="True Positive Rate"),
        color=alt.Color('name', legend=alt.Legend(
            orient='bottom-right',
            title='',
            fillColor='white',
            titleAnchor='middle'))
    ).properties(title='ROC Curves')

    diag_line_df = pd.DataFrame({'var1': [0, 1], 'var2': [0, 1]})
    diag_line = alt.Chart(diag_line_df).mark_line(color='black', strokeWidth=2, strokeDash=[1, 1]).encode(
        x='var1',
        y='var2')
    roc_curves = roc_curves + diag_line

    return pr_curves, roc_curves


def plot_permutation_imp(model_fit, validation_data, scoring="log_loss", title=''):
    # if scoring == 'log_loss':
    #     log_loss_scorer = make_scorer(log_loss, greater_is_better=False)

    val_dataset = DataSet(model_fit['components']['DataSet']['config'], validation_data)

    X = val_dataset.x
    y = val_dataset.y
    print(1)
    result = permutation_importance(model_fit, X, y, n_repeats=10, scoring=log_loss_scorer,
                                    random_state=42, n_jobs=1)
    #
    # sorted_idx = result.importances_mean.argsort()
    # # fig, ax = plt.subplots()
    # # plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
    # # ax.set_title("Permutation Importance {}".format(title))
    # # fig.tight_layout()
    # results_df = pd.DataFrame(result.importances[sorted_idx].T, columns=X.columns[sorted_idx]).T.reset_index()
    # print(2)
    #
    # results_df_alt = pd.melt(results_df, value_name="Importances", id_vars="index")
    # c = alt.Chart(results_df_alt).mark_boxplot(extent='min-max').encode(
    #     x=alt.X('Importances', sort='-x'),
    #     y='index',
    # )
    # print(3)

    return None
