class Predictor:

    def __init__(self, model=None):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        """Enforce returning only a single series."""
        y_pred_proba = self.model.predict_proba(x)
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        return y_pred_proba