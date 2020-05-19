import altair as alt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import pandas as pd
from IPython.display import display

def evaluations_model_test_set(test_set, predictor):
    '''Takes care of all LR model evaluations in test set'''
    test_true_labels = test_set['NoShow'].copy()
    test_true_labels = test_true_labels.to_numpy()

    # Numpy array with all predictions
    prediction_classes = predictor.predict(test_set)

    # Evaluation model
    print(classification_report(test_true_labels, prediction_classes))

    # EVALUATION MODEL - plot of the ROC Curve
    falsepositive_r, truepositive_r, roc_auc = roc_curve(test_true_labels,prediction_classes)

    source = pd.DataFrame({
        'False Positive Rate': falsepositive_r,
        'True Positive Rate': truepositive_r
    })
    source['Curve'] = 'ROC'
    source = source.append({'False Positive Rate': 0, 'True Positive Rate': 0, 'Curve':'Random'}, ignore_index=True)
    source = source.append({'False Positive Rate': 1, 'True Positive Rate': 1, 'Curve':'Random'}, ignore_index=True)
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

    ## Evaluating model - Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    print('The ROC_AUC_SCORE is : {}'.format(roc_auc_score(test_true_labels, prediction_classes)))