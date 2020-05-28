import altair as alt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import pandas as pd
from IPython.display import display
from typing import Any, Callable


def evaluate_model_on_test_set(true_labels: Any, predicted_labels: Any, label_name='NoShow'):
    """
    Prints a classification report, confusion matrix, and ROC curve for a predictor.
    """
    # Evaluation model
    print(classification_report(true_labels, predicted_labels))

    # Evaluation Model - plot of the ROC Curve
    falsepositive_r, truepositive_r, roc_auc = roc_curve(true_labels, predicted_labels)

    source = pd.DataFrame({
        'False Positive Rate': falsepositive_r,
        'True Positive Rate': truepositive_r
    })
    source['Curve'] = 'ROC'
    source = source.append({'False Positive Rate': 0, 'True Positive Rate': 0, 'Curve': 'Random'}, ignore_index=True)
    source = source.append({'False Positive Rate': 1, 'True Positive Rate': 1, 'Curve': 'Random'}, ignore_index=True)
    print(source.head())

    chart = alt.Chart(source).mark_line().encode(
        x='False Positive Rate',
        y='True Positive Rate',
        strokeDash='Curve',
        color='Curve'
    ).properties(
        title='Receiver operating characteristic'
    )
    display(chart)

    # Evaluating model - Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    print('The ROC_AUC_SCORE is : {}'.format(roc_auc_score(true_labels, predicted_labels)))
