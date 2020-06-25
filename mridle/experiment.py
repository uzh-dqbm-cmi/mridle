import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from typing import Any, Dict, List, Tuple, Callable
from sklearn.preprocessing import StandardScaler


class ModelRun:
    """Base class for a model experiment run.

    To use this class, subclass this class and fill out the following placeholder functions with the specific
    implementation of the model experiment:
     - `build_x_features` (required)
     - `train_encoders` (optional)
     - `build_y_vector` (optional)
    """

    def __init__(self, train_set: Any, test_set: Any, label_key: str, model: Any, hyperparams: Dict,
                 preprocessing_func: Callable, feature_subset: List = None, reduce_features=False):
        """
        Create a ModelRun object.

        Args:
            train_set: The data to train the model on. May be in any format, as long as `self.build_x_features` can work
             with it. Includes both the input and target of the model.
            test_set: The data to test and evaluate the model on. Must be in the same format as `train_set`.
            label_key: A string that is passed to `self.build_y_vector` that is used to identify the label column or key
             in the train and test sets.
            model:
            hyperparams: Dictionary of hyperparameter options to try with sklearn.model_selection.RandomizedSearchCV.
            preprocessing_func: The function that was used to transform the raw_data into train_set. Is saved to pass to
             Predictor to maintain and end-to-end record of the the history of data the model was trained on.
            feature_subset: List of str, or None if NA. Subset of features to use. If this value is set,
             `self.build_x_features` will restrict the X datasets to the specified columns.
            reduce_features: Whether to use sklearn.feature_selection.RFECV to identify a subset of features.
        """
        self.train_set = train_set
        self.test_set = test_set
        self.label_key = label_key
        self.encoders = {}
        self.model = clone(model)
        self.hyperparams = hyperparams
        self.feature_subset = feature_subset
        self.reduce_features = reduce_features

        # properties that will be set during self.run()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_predicted = None
        self.y_test_predicted = None
        self.feature_cols = []
        self.evaluation = None
        self.rfecv = None

        # saved to maintain and end-to-end record of the the history of data the model was trained on
        self.preprocessing_func = preprocessing_func

    def run(self, run_hyperparam_search: bool = True) -> Dict:
        """
        Run the modeling experiment.

        Args:
            run_hyperparam_search: Whether to do a hyperparameter search as part of the modeling experiment.

        Returns: Model evaluation dictionary.

        """
        self.x_train, self.x_test, self.y_train, self.y_test, self.feature_cols, self.encoders = self.build_data(
            self.train_set, self.test_set, self.label_key, self.feature_subset)

        # TODO: refactor this section. LKK note: this many if/elses is no good at all!
        if run_hyperparam_search:
            self.model = self.search_hyperparameters(self.model, self.hyperparams, self.x_train, self.y_train)
            if self.reduce_features:
                self.rfecv = self.train_feature_reducer(self.model, self.x_train, self.y_train)
                self.model = self.rfecv.estimator_
                self.feature_cols = self.rebuild_feature_cols_from_rfecv(self.feature_cols, self.rfecv.support_)
        if self.reduce_features:
            self.y_train_predicted = self.rfecv.predict(self.x_train)
            self.y_test_predicted = self.rfecv.predict(self.x_test)
            self.evaluation = self.evaluate_model(self.rfecv, self.x_test, self.y_test, self.y_test_predicted)
        else:
            self.model.fit(self.x_train, self.y_train)
            self.y_train_predicted = self.model.predict(self.x_train)
            self.y_test_predicted = self.model.predict(self.x_test)
            self.evaluation = self.evaluate_model(self.model, self.x_test, self.y_test, self.y_test_predicted)
        return self.evaluation

    @classmethod
    def build_data(cls, train_set: Any, test_set: Any, label_key: str, feature_subset: List[str]) -> \
            Tuple[pd.DataFrame, pd.DataFrame, List, List, List[str], Dict[str, Any]]:
        """Orchestrates the construction of train and test x matrices, and train and test y vectors.

        `build_data` takes as input:
            - train_set: Any
            - test_set: Any
            - label_key: str. key to use in data dicts for label
            - feature_subset: List of str, or None if NA. Subset of features to use.

        `build_data` returns a Tuple of the following:
            - x_train: pd.DataFrame
            - x_test: pd.DataFrame
            - y_train: List
            - y_test: List
            - feature_cols: List[str]
            - encoders: Dict[str, Any]. Encoders used to generate the feature set. Encoders that may want to be saved
                include vectorizers trained on the train_set and applied to the test_set.

        """

        encoders = cls.train_encoders(train_set)

        x_train, feature_cols = cls.build_x_features(train_set, encoders)
        x_test, feature_cols = cls.build_x_features(test_set, encoders)

        if feature_subset:
            x_train = cls.restrict_features_to_subset(x_train, feature_subset)
            x_test = cls.restrict_features_to_subset(x_test, feature_subset)
            feature_cols = feature_subset

        y_train = cls.build_y_vector(train_set, label_key)
        y_test = cls.build_y_vector(test_set, label_key)

        return x_train, x_test, y_train, y_test, feature_cols, encoders

    @classmethod
    def train_encoders(cls, train_set: Any) -> Dict[str, Any]:
        """
        Placeholder function to hold the custom encoder training functionality of a ModelRun.
        By default, returns an empty dictionary.

        Args:
            train_set: Data set to train encoders on.

        Returns:
            Dict of encoders.
        """
        autoscaler = StandardScaler()
        features = 'historic_no_show_cnt'
        train_set[features] = autoscaler.fit_transform(train_set[features])

        return {
            'autoscaler': autoscaler
        }

    @classmethod
    def build_x_features(cls, data_set: Any, encoders: Dict) -> pd.DataFrame:
        """
        Placeholder function to hold the custom feature building functionality of a ModelRun.

        Args:
            data_set: Data set to transform into features.
            encoders: Dict of pre-trained encoders for use in building features.

        Returns:
            Matrix-type
        """
        raise NotImplementedError("The ModelRun class must be subclassed to be used, "
                                  "with the `build_x_features` function implemented.")

    @classmethod
    def build_y_vector(cls, data_set: Any, label_key: str) -> List:
        """
        Extract the labels from the data set.
        This base implementation assumes that `data_set` is a pd.DataFrame and that label_key refers to a column in the
          data_set. If this is not the structure of the model, overwrite this method.

        Args:
            data_set: Tither training or testing data.
            label_key: The key in the data under which the label is stored.

        Returns:
            Array-type
        Raises:
            ValueError if label_key not found.
            NotImplementedError if data_set is not a pd.DataFrame.
        """
        if type(data_set) == pd.DataFrame:
            if label_key in data_set.columns:
                return data_set[label_key].copy()
            else:
                raise ValueError('label_key {} not found in the data_set.'.format(label_key))
        else:
            raise NotImplementedError('The base implementation of build_y_vector expects a pd.DataFrame, but received a'
                                      ' {}. To resolve this, overwrite the base implementation of build_y_vector.'
                                      ''.format(type(data_set)))

    @classmethod
    def restrict_features_to_subset(cls, df: pd.DataFrame, feature_subset: List[str]) -> pd.DataFrame:
        """
        Restrict the data to only the columns in feature_subset.
        If feature_subset includes columns not present in the df, they are ignored.

        Args:
            df: Dataframe to subset.
            feature_subset: List of columns to restrict df to.

        Returns: Subset of df.

        """
        column_mask = [col in feature_subset for col in df.columns]
        df = df[column_mask].copy()
        df.columns = feature_subset
        return df

    @classmethod
    def search_hyperparameters(cls, model: Any, hyperparams: Dict[str, List], x_train: pd.DataFrame,
                               y_train: List) -> Any:
        """
        Run sklearn.model_selection.Randomized_SearchCV and return the best model.
        Args:
            model: The model to select hyperparameters for.
            hyperparams: Dictionary of hyperparameter options.
            x_train: Training data input.
            y_train: Training data labels.

        Returns: The best model.

        """
        random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparams, n_iter=5, cv=2, verbose=2,
                                           random_state=42, n_jobs=-1, scoring='f1_macro')
        random_search.fit(x_train, y_train)
        return random_search.best_estimator_

    @classmethod
    def train_feature_reducer(cls, model, x_train, y_train) -> RFECV:
        """
        Train the sklearn.feature_selection.RFECV.

        Args:
            model: The model.
            x_train: training data
            y_train: training data labels

        Returns: rfecv object

        """
        rfecv = RFECV(estimator=model, step=0.05, scoring='f1_macro', n_jobs=-1)  # , cv=StratifiedKFold)
        rfecv.fit(x_train, y_train)
        print("Optimal number of features : %d" % rfecv.n_features_)
        print("Params: {}".format(rfecv.get_params()))
        return rfecv

    @classmethod
    def rebuild_feature_cols_from_rfecv(cls, feature_cols, support):
        new_feature_cols = []
        for i, val in enumerate(support):
            if val:
                new_feature_cols.append(feature_cols[i])
        return new_feature_cols

    @classmethod
    def evaluate_model(cls, model: Any, x_test: pd.DataFrame, y_test: List, y_test_predicted: List) -> Dict[str, Any]:
        """
        Calculate and return a dictionary of various evaluation metrics.

        Args:
            model: a model with a `score` method.
            x_test: input of the test set.
            y_test: true labels for the test set.
            y_test_predicted: the model's predictions for the test set.

        Returns:

        """
        accuracy = model.score(x_test, y_test)
        f1_macro = f1_score(y_test, y_test_predicted, average='macro')
        f1_micro = f1_score(y_test, y_test_predicted, average='micro')
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Generate the feature importances of the trained model. Requires self.model to have a `feature_importances_`
        member variable.

        Returns: pd.DataFrame of feature importances in descending order.

        """
        if hasattr(self.model, 'feature_importances_'):
            fi = pd.DataFrame(self.model.feature_importances_, index=self.feature_cols)
            fi = fi.sort_values(0, ascending=False)
            return fi
        else:
            return pd.DataFrame()

    def get_hyperparams(self, name: str = None) -> pd.DataFrame:
        """
        Get the hyperparameters of the model.
        Getting hyperparameters for various sklearn models is done differently, so ModelRun contains custom functions
         for retrieiving them (in model_hyperparam_func_map).

        Args:
            name: The name of the model for labeling the index of the resulting dataframe.

        Returns: Summary of model hyperparameters in a pd.DataFrame.

        Raises: NotImplementedError if the model is a type that is not registered in model_hyperparam_func_map.

        """
        model_hyperparam_func_map = {
            "<class 'sklearn.ensemble._forest.RandomForestClassifier'>": self.get_selected_random_forest_hyperparams,
            "<class 'sklearn.svm._classes.SVC'>": self.get_selected_svc_hyperparams,
            "<class 'sklearn.linear_model._logistic.LogisticRegression'>": self.get_selected_logistic_reg_hyperparams
        }
        model_type = str(type(self.model))
        if model_type in model_hyperparam_func_map:
            model_hyperparam_func = model_hyperparam_func_map[model_type]
            chosen_hyperparams = model_hyperparam_func(self.model)
            chosen_hyperparams['num_features'] = len(self.feature_cols)

            if name is None:
                name = 'model'

            return pd.DataFrame(chosen_hyperparams, index=[name])
        else:
            raise NotImplementedError('There is no hyperparameter retreival function implemented in ModelRun for {}'
                                      ' models'.format(model_type))

    @classmethod
    def get_selected_random_forest_hyperparams(cls, model: Any) -> Dict[str, Any]:
        """Get hyperparams out of a sklearn random forest model. """

        chosen_hyperparams = {
            'n_estimators': model.n_estimators,
            'max_features': model.max_features,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'class_weight': model.class_weight,
        }
        return chosen_hyperparams

    @classmethod
    def get_selected_svc_hyperparams(cls, model: Any) -> Dict:
        """Get hyperparams out of a sklearn SVC model. """

        chosen_hyperparams = model.get_params()
        return chosen_hyperparams

    @classmethod
    def get_selected_logistic_reg_hyperparams(cls, model: Any) -> Dict:
        """Get hyperparams out of a sklearn LogisticRegression model. """

        chosen_hyperparams = model.get_params()
        return chosen_hyperparams

    def generate_predictor(self) -> 'Predictor':
        """
        Return a Predictor object from the trained model.

        Returns:
            Predictor
        """
        return Predictor(self.model, self.encoders, self.preprocessing_func, self.build_x_features)


class Predictor:

    def __init__(self, model: Any, encoders: Dict[str, Any], preprocess_func: Callable, transform_func: Callable):
        """
        Base class for a machine learning model predictor.

        Args:
            model: A trained model with a `predict` method (such as a sklearn model).
            encoders: Dictionary of encoders to be used during feature transformation.
            transform_func: Function to transform input data into features for the model.
        """
        self.model = model
        self.preprocess_fun = preprocess_func
        self.encoders = encoders
        self.transform_func = transform_func

    def predict(self, data_point: Any) -> Any:
        """
        Make a prediction.

        Args:
            data_point: Raw input data point to make a prediction for.

        Returns: prediction value.

        """
        data_processed = self.preprocess_fun(data_point)
        x, feature_cols = self.transform_func(data_processed, self.encoders)
        return self.model.predict(x), self.model.predict_proba(x)
