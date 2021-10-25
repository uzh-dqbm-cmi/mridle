import datetime
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from typing import Any, Dict, List, Tuple, Callable
from hyperopt import fmin, tpe, Trials, space_eval
from functools import partial
from sklearn.metrics import brier_score_loss, log_loss, f1_score


class ModelRun:
    """Base class for a model experiment run.

    This class may be used as is, or subclassed for a more customized modeling workflow.
    This base class assumes that train_set is a pd.DataFrame.

    To customized the behavior of ModelRun, subclass this class and modify/fill out the following  functions with the
     specific implementation of your model workflow:
     - `build_x_features`
     - `train_encoders`
     - `build_y_vector`

     To test your ModelRun implementation, you can implement `get_test_data_set` to provide a dataset for use in tests.
    """

    def __init__(self, train_set: Any, test_set: Any, label_key: str, model: Any, hyperparams: Dict, search_type: str,
                 num_cv_folds: int, num_iters: int, scoring_fn: str, preprocessing_func: Callable,
                 feature_subset: List = None, reduce_features=False):
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
            search_type: Type of search to do when searching for hyperparameters (grid, random, or bayesian)
            num_cv_folds: Number of cross validation folds to run in the hyperparameter search
            num_iters: Number of iterations to run per folds in the hyperparameter search (only applicable for
             search_type='random')
            scoring_fn: the scoring function to use (can be from 'f1_macro', 'log_loss', or 'brier_score')
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
        self.hyperopt_trials = Trials()
        self.search_type = search_type
        self.num_cv_folds = num_cv_folds
        self.num_iters = num_iters
        self.scoring_fn = scoring_fn

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

        # Will be set when model is saved, and then re-used if model is run further (for hyperparam search)
        self.file_path = None

        # Create datasets
        self.x_train, self.x_test, self.y_train, self.y_test, self.encoders = self.build_data(
            self.train_set, self.test_set, self.label_key, self.feature_subset)

    def run(self, run_hyperparam_search: bool = True, hyperopt_timeout: int = 360) -> Dict:
        """
        Run the modeling experiment.

        Args:
            run_hyperparam_search: Whether to do a hyperparameter search as part of the modeling experiment.
            hyperopt_timeout: If running hyperopt search, the user can specify how long to run this for (in seconds).

        Returns: Model evaluation dictionary.

        """

        # TODO: refactor this section. LKK note: this many if/elses is no good at all!
        if run_hyperparam_search:
            self.model = self.search_hyperparameters(self.model, self.hyperparams, self.x_train, self.y_train,
                                                     search_type=self.search_type, num_cv_folds=self.num_cv_folds,
                                                     num_iters=self.num_iters, hyperopt_timeout=hyperopt_timeout,
                                                     hyperopt_trials=self.hyperopt_trials, scoring_fn=self.scoring_fn)
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
            Tuple[pd.DataFrame, pd.DataFrame, List, List, Dict[str, Any]]:
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
        x_train = cls.build_x_features(train_set, feature_subset, label_key, encoders)
        x_test = cls.build_x_features(test_set, feature_subset, label_key, encoders)

        y_train = cls.build_y_vector(train_set, label_key)
        y_test = cls.build_y_vector(test_set, label_key)

        return x_train, x_test, y_train, y_test, encoders

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
        return {}

    @classmethod
    def get_test_data_set(cls):
        """Provide a small dataset for use in test cases. Dataset should be small, and have the same column names and
        dtypes as is expected of the real input data."""
        raise NotImplementedError("The ModelRun class must be subclassed to be used, "
                                  "with the `build_x_features` function implemented.")

    @classmethod
    def build_x_features(cls, data_set: Any, feature_subset: List[str], label_key: str = '', encoders: Dict = None
                         ) -> pd.DataFrame:
        """
        Create the X feature set from the data set by removing the label column.

        Args:
            data_set: Data set to transform into features.
            feature_subset: Subset of features to use in X data
            label_key: Name of the label column that will be removed from the dataset to generate the feature set.
            encoders: Dict of pre-trained encoders for use in building features.

        Returns:
            Tuple containing the pd.DataFrame of the feature set and a list of the column names.
        """
        data_subset = data_set.copy()
        if feature_subset:
            data_subset = data_subset[feature_subset]
        data_subset = data_subset.drop([label_key], axis=1, errors='ignore')
        return data_subset

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
                               y_train: List, scoring_fn: str, search_type: str, num_cv_folds: int, num_iters: int,
                               hyperopt_timeout: int, hyperopt_trials: Any) -> Any:
        """
        Run sklearn.model_selection.Randomized_SearchCV and return the best model.
        Args:
            model: The model to select hyperparameters for.
            hyperparams: Dictionary of hyperparameter options.
            x_train: Training data input.
            y_train: Training data labels.
            scoring_fn: the scoring function to use (can be from 'f1_macro', 'log_loss', or 'brier_score')
            search_type: Type of search for hyperparameters. Choose between random, grid, and bayesian search. All
             search types include cross validation
            num_cv_folds: Number of cross validation folds to run.
            num_iters: Number of iterations to run (only applicable for random search)
            hyperopt_timeout: If running hyperopt search, the user can specify how long to run this for (in seconds).
            hyperopt_trials: If running hyperopt search, the Trials object holds the results of previous hyperparameter
             evaluations, and uses these to guide the future search.

        Returns: The best model.

        """
        if search_type == "random":
            random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparams, n_iter=num_iters,
                                               cv=num_cv_folds, verbose=2, random_state=42, n_jobs=-1,
                                               scoring=scoring_fn)
            random_search.fit(x_train, y_train)
            best_est = random_search.best_estimator_
        elif search_type == "grid":
            grid_search = GridSearchCV(estimator=model, param_grid=hyperparams, cv=num_cv_folds, verbose=2,
                                       n_jobs=-1, scoring=scoring_fn)
            grid_search.fit(x_train, y_train)
            best_est = grid_search.best_estimator_
        elif search_type == "bayesian":
            best_est = cls.bayesian_param_search(model, hyperparams, x_train, y_train, scoring_fn=scoring_fn,
                                                 trials=hyperopt_trials, timeout=hyperopt_timeout, nfolds=num_cv_folds)

        else:
            raise NotImplementedError(
                'search_type should be one of ''random'', ''grid'' or ''bayesian. ''{}'' given'.format(search_type))

        return best_est

    @classmethod
    def bayesian_param_search(cls, model, hyperparameters, x_train, y_train, scoring_fn, trials, timeout=5 * 360,
                              max_evals=150, nfolds=5, print_result=True):
        """
        Function which performs the full Bayesian hyperparameter search. Uses hyperopt package as the base, and our own
        hyperopt_objective() function as a helper.

        This function takes in a model, data, a scoring function - similar to other functions.

        Importantly, it also requires a set of hyperparameter distributions (defined using the hyperopt format) which it
        searches over. It performs this search until either max_evals is reached, or the time limit (timeout) is passed.
        The results of these trials is saved to the provied trials object, which is itself an attribute of the Modelrun
        class.

        Args:
            model: model
            hyperparameters: hyperparam space which the function is to search over. Defined using hyperopt format, as in
            the following link:  http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
            x_train: training data
            y_train: labels for training data
            scoring_fn: the scoring function to use (can be from 'f1_macro', 'log_loss', or 'brier_score')
            trials: hyperopt.Trials() object, used to save the results from the search
            timeout: time (in seconds) to run the search for
            max_evals: number of evaluations / iterations to make in the search, before ending
            nfolds: number of folds to use in cross validation
            print_result: boolean, giving user preference of whether to print information as the trials are being run

        Returns:
            A model fit on the provided data, using the 'best' hyperparams as found by Bayesian optimisation
        """
        space = hyperparameters

        cv_ids = list(range(nfolds)) * np.floor((len(x_train) / nfolds)).astype(int)
        cv_ids.extend(list(range(len(x_train) % nfolds)))
        cv_ids = np.random.permutation(cv_ids)

        best_rf = fmin(partial(cls.hyperopt_objective, model=model, x_train=x_train, y_train=y_train,
                               scoring_fn=scoring_fn, ids=cv_ids, nfolds=nfolds, print_result=print_result),
                       space, algo=tpe.suggest, timeout=timeout, max_evals=max_evals, trials=trials)
        best_params = space_eval(space, best_rf)
        model = model.set_params(**best_params)

        return model.fit(x_train, y_train)

    @classmethod
    def hyperopt_objective(cls, params, model, x_train, y_train, scoring_fn: str, ids: List[int], nfolds, print_result):
        """
        Objective to minimise. For use with the hyperopt package, which performs Bayesian hyperparameter searches.
        This takes in the model, data, and a list of parameter values that should be used for calculating the loss

        Args:
            params: the parameter set to test and calculate the cross validated loss for
            model: the model
            x_train: training data
            y_train: training data labels
            scoring_fn: the scoring function to use (can be from 'f1_macro', 'log_loss', or 'brier_score')
            ids: list of ints, the same length as x_train, which holds information on which CV fold each row should be
            assigned to
            nfolds: number of folds to use in cross validation
            print_result: boolean, giving user preference of whether to print information as the trials are being run

        Returns:
            Loss associated with the given parameters, which is to be minimised over time.

        """

        model = model
        model = model.set_params(**params)

        cv_results = []
        for k in range(nfolds):
            x_train_cv = x_train[ids != k]
            y_train_cv = y_train[ids != k]
            x_test_cv = x_train[ids == k]
            y_test_cv = y_train[ids == k]

            model = model.fit(x_train_cv, y_train_cv)

            if scoring_fn == 'f1_macro':
                preds = model.predict(x_test_cv)
                loss = -1 * f1_score(y_test_cv, preds, average='macro')
            elif scoring_fn == 'log_loss':
                probs = model.predict_proba(x_test_cv)[:, 1]
                loss = log_loss(y_test_cv, probs)
            elif scoring_fn == 'brier_score':
                probs = model.predict_proba(x_test_cv)[:, 1]
                loss = brier_score_loss(y_test_cv, probs)
            else:
                raise NotImplementedError(
                    'scoring_fn should be one of ''f1_macro'', ''log_loss'', or ''brier_score''. ''{}'' given'.format(
                        scoring_fn
                    ))

            cv_results.append(loss)

        to_minimise = np.mean(cv_results)
        if print_result:
            print(params)
            print('Loss: {}'.format(to_minimise))

        return to_minimise

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
            "f1_micro": f1_micro
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Generate the feature importances of the trained model. Requires self.model to have a `feature_importances_`
        member variable.

        Returns: pd.DataFrame of feature importances in descending order.

        """
        if hasattr(self.model, 'feature_importances_'):
            fi = pd.DataFrame(self.model.feature_importances_, index=self.feature_subset)
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
            "<class 'sklearn.svm._classes.SVC'>": self.get_standard_sklearn_params,
            "<class 'sklearn.linear_model._logistic.LogisticRegression'>": self.get_standard_sklearn_params,
            "<class 'xgboost.sklearn.XGBClassifier'>": self.get_standard_sklearn_params,
        }
        model_type = str(type(self.model))
        if model_type in model_hyperparam_func_map:
            model_hyperparam_func = model_hyperparam_func_map[model_type]
            chosen_hyperparams = model_hyperparam_func(self.model)
            chosen_hyperparams['num_features'] = len(self.feature_subset)

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
    def get_standard_sklearn_params(cls, model: Any) -> Dict:
        """Get hyperparams out of a standard sklearn model. """

        chosen_hyperparams = model.get_params()
        return chosen_hyperparams

    def generate_file_name(self, descriptor: str = None):
        """
        Generate a filename for a model that includes the timestamp, model type, and an optional descriptor.
        These properties are separated by '__' and the filename ends in .pkl.

        Args:
            descriptor: Optional descriptor to add to the file name.

        Returns:
            File name with file extension.
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_type = self.model.__class__.__name__
        delimiter_char = '__'
        file_name_components = [timestamp, model_type]
        if descriptor is not None:
            descriptor = descriptor.replace(' ', '-')
            file_name_components.append(descriptor)
        file_name = delimiter_char.join(file_name_components) + '.pkl'
        return file_name

    def save(self, parent_directory: str, descriptor: str = None) -> Path:
        """
        Save a model as a pickle to a parent_directory with a programmatic filename that includes a timestamp,
         model type, and optional descriptor.

        Args:
            parent_directory: The parent directory in which to save the model.
            descriptor: Optional descriptor to add to the file name.

        Returns:
            File path of the saved object.
        Example Usage:
            >>> my_model_run.save('project/data/models/')
            >>> # saves project/data/models/YYYY-MM-DD_HH-MM-SS__<model_class>.pkl
            >>> my_model_run.save('project/data/models/', descriptor='5 features')
            >>> # saves project/data/models/YYYY-MM-DD_HH-MM-SS__<model_class>__5-features.pkl
        """
        if self.file_path:
            with open(self.file_path, 'wb+') as f:
                pickle.dump(self, f)
        else:
            file_name = self.generate_file_name(descriptor)
            self.file_path = Path(parent_directory, file_name)
            with open(self.file_path, 'wb+') as f:
                pickle.dump(self, f)
        return self.file_path

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
        x, feature_subset = self.transform_func(data_processed, self.encoders)
        return self.model.predict(x), self.model.predict_proba(x)


class PartitionedExperiment:
    """
    Run a ModelRun experiment multiple times on different partitions of the data.
    Partitions the data_set into <n_partition> sets (by default, stratifies by label), and runs the ModelRun on each
     partition. Summarizes metric results across partitions via `show_evaluation()`and feature_importances via
     `show_feature_importances()`.
    """

    def __init__(self, name: str, data_set: Any, label_key: str, preprocessing_func: Callable,
                 model_run_class: ModelRun, model, hyperparams: Dict, search_type: str, num_cv_folds: int,
                 num_iters: int, scoring_fn: str, n_partitions: int = 5, stratify_by_label: bool = True,
                 feature_subset=None, reduce_features=False, verbose=False):
        """

        Args:
            name: Name for identification of the Experiment.
            data_set: Data to run the model on.
            label_key: the key to use to calculate the label.
            preprocessing_func: the function that was used to preprocess `data_set`.
                Passed through to ModelRun in order to be able to generate Predictor objects.
            model_run_class: An implemented subclass of ModelRun.
            model: A model with a fit() method.
            hyperparams: Dictionary of hyperparamters to search for the best model.
            search_type: Type of search to do when searching for hyperparameters (grid, random, or bayesian)
            num_cv_folds: Number of cross validation folds to run in the hyperparameter search
            num_iters: Number of iterations to run per folds in the hyperparameter search (only applicable for
             search_type='random')
            scoring_fn: the scoring function to use (can be from 'f1_macro', 'log_loss', or 'brier_score')
            n_partitions: Number of partitions to split the data on and run the experiment on.
            stratify_by_label: Whether to stratify the partitions by label (default), otherwise partition randomly.
            feature_subset: Subset of features to use. If not used, None is passed.
            reduce_features: Whether to use recursive feature elimination to reduce features.
            verbose: Whether to display verbose messages.
        """

        if n_partitions < 2:
            raise ValueError("n_partitions must be greater than 1. Got {}.".format(n_partitions))

        self.name = name
        self.data_set = data_set
        self.label_key = label_key
        self.preprocessing_func = preprocessing_func
        self.verbose = verbose
        self.model_run_class = model_run_class
        self.model = model
        self.hyperparams = hyperparams
        self.stratify_by_label = stratify_by_label
        self.feature_subset = feature_subset
        self.reduce_features = reduce_features
        self.hyperopt_trials = Trials()
        self.search_type = search_type
        self.num_cv_folds = num_cv_folds
        self.num_iters = num_iters
        self.scoring_fn = scoring_fn

        self.n_partitions = n_partitions

        if stratify_by_label:
            self.partition_ids = self.partition_data_stratified(data_set[label_key], self.n_partitions)
        else:
            self.partition_ids = self.partition_data_randomly(self.n_partitions)

        self.model_runs = {}
        self.all_run_results = []
        self.experiment_results = []

        if verbose:
            print("Partition Stats for {}".format(self.name))
            self.report_partition_stats(self.partition_ids, data_set, label_key)

    def run(self, num_partitions_to_run=None, run_hyperparam_search=True,
            hyperopt_timeout: int = 360) -> List[Dict[str, Any]]:
        """
        Run the experiment on all partitions.

        Args:
            num_partitions_to_run: select a subset of partitions to run, for faster testing.
            run_hyperparam_search: argument to turn off hyperparam search, for faster testing.
            hyperopt_timeout: If running hyperopt search, the user can specify how long to run this for (in seconds).

        Returns: List of experiment results (dicts from ModelRun.evaluation) for each of the partitions.

        """
        partitions_to_run = list(self.partition_ids.keys())
        if num_partitions_to_run is not None:
            partitions_to_run = partitions_to_run[:num_partitions_to_run]
            print("Running only partitions {}".format(", ".join(partitions_to_run)))

        for partition_name in self.partition_ids:
            if partition_name in partitions_to_run:
                print("Running partition {}...".format(partition_name))
                model_run = self.run_experiment_on_one_partition(data_set=self.data_set, label_key=self.label_key,
                                                                 partition_ids=self.partition_ids[partition_name],
                                                                 preprocessing_func=self.preprocessing_func,
                                                                 model_run_class=self.model_run_class,
                                                                 model=self.model,
                                                                 hyperparams=self.hyperparams,
                                                                 run_hyperparam_search=run_hyperparam_search,
                                                                 search_type=self.search_type,
                                                                 num_cv_folds=self.num_cv_folds,
                                                                 num_iters=self.num_iters,
                                                                 scoring_fn=self.scoring_fn,
                                                                 feature_subset=self.feature_subset,
                                                                 reduce_features=self.reduce_features,
                                                                 hyperopt_timeout=hyperopt_timeout)
                self.model_runs[partition_name] = model_run

        print("Compiling results")
        self.experiment_results = self.summarize_runs(self.model_runs)
        return self.experiment_results

    @classmethod
    def run_experiment_on_one_partition(cls, data_set: Dict, label_key: str, partition_ids: List[int],
                                        preprocessing_func: Callable, model_run_class: ModelRun, model,
                                        hyperparams: Dict, run_hyperparam_search: bool, search_type, scoring_fn: str,
                                        num_cv_folds: int, num_iters: int, feature_subset: List[str],
                                        reduce_features: bool, hyperopt_timeout: int = 60):
        train_set, test_set = cls.materialize_partition(partition_ids, data_set)
        mr = model_run_class(train_set=train_set, test_set=test_set, label_key=label_key, model=model,
                             preprocessing_func=preprocessing_func, hyperparams=hyperparams, search_type=search_type,
                             num_cv_folds=num_cv_folds, num_iters=num_iters, scoring_fn=scoring_fn,
                             feature_subset=feature_subset, reduce_features=reduce_features)
        mr.run(run_hyperparam_search=run_hyperparam_search, hyperopt_timeout=hyperopt_timeout)
        return mr

    @classmethod
    def partition_data_randomly(cls, data_set: pd.DataFrame, n_partitions: int) -> Dict[str, List[int]]:
        """Randomly shuffle and split the data set into n roughly equal partitions."""
        raise NotImplementedError

    @classmethod
    def partition_data_stratified(cls, label_list: List[int], n_partitions: int) -> \
            Dict[str, Tuple[List[int], List[int]]]:
        """Randomly shuffle and split the doc_list into n roughly equal lists, stratified by label."""
        skf = StratifiedKFold(n_splits=n_partitions, random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = skf.split(x, label_list)
        partitions = {}
        for p_id, p in enumerate(partition_indexes):
            partitions['Partition {}'.format(p_id)] = p
        return partitions

    @classmethod
    def materialize_partition(cls, partition_ids: Tuple[List[int], List[int]], data_set: pd.DataFrame)\
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training and testing dataset based on the partition, which indicate the ids for the test set.

        Args:
            partition_ids: Dictionary, where the values contain the indices of each partition's test set.
            data_set: The full data set.

        Returns: Train set and Test set.
        """
        train_partition_ids, test_partition_ids = partition_ids
        train_set = data_set.iloc[train_partition_ids]
        test_set = data_set.iloc[test_partition_ids]
        return train_set, test_set

    @classmethod
    def report_partition_stats(cls, partition_ids: Dict[str, Tuple[List[int], List[int]]], data_set: Any,
                               label_key=str):
        """
        Print the size and class balance of each partition's train and test set.

        Args:
            partition_ids: Partitions.
            data_set: The full data set.
            label_key: The column name in data_set that is the label.

        Returns: None, just prints.
        """
        for partition_name in partition_ids:
            p_ids = partition_ids[partition_name]
            train_set, test_set = cls.materialize_partition(p_ids, data_set)

            print('\n-Partition {}-'.format(partition_name))
            print("Train: {:,.0f} data points".format(len(train_set)))
            print("Test: {:,.0f} data points".format(len(test_set)))

            label_options = data_set[label_key].unique()
            for label_i in label_options:
                data_set_dict = {'Train': train_set, 'Test': test_set}
                for data_subset_name in data_set_dict:
                    data_subset = data_set_dict[data_subset_name]
                    pct_label_i = data_subset[data_subset[label_key] == label_i].shape[0] / data_subset.shape[0]
                    print("{} Set: {:.0%} {}".format(data_subset_name, pct_label_i, label_i))

    @classmethod
    def summarize_runs(cls, run_results: Dict) -> List[Dict[str, Any]]:
        """Compile the ModelRun.evaluation dictionaries from all partitions into a list."""
        return [run_results[mr].evaluation for mr in run_results]

    def show_feature_importances(self):
        """
        Build a table of feature importances for each partition.

        Returns: Dataframe with features as rows partitions as columns. Dataframe is sorted by highest median feature
         importance.
        """
        all_feature_importances = pd.DataFrame()
        for partition_id in self.model_runs:
            partition_feature_importances = self.model_runs[partition_id].get_feature_importances()

            # if the model has no feature importances, just exit now
            if partition_feature_importances.shape[0] == 0:
                return pd.DataFrame()

            partition_feature_importances.columns = [partition_id]
            all_feature_importances = pd.merge(all_feature_importances, partition_feature_importances, how='outer',
                                               left_index=True, right_index=True)
        all_feature_importances['median'] = all_feature_importances.median(axis=1)
        return all_feature_importances.sort_values('median', ascending=False)

    def get_hyperparams(self) -> pd.DataFrame:
        """
        Combines all hyperparamters from each of the partitions.

        Returns: pd.DataFrame of all hyperparameters and their values.
        """
        all_hyperparams = pd.DataFrame()
        for partition_id in self.model_runs:
            partition_hyperparams = self.model_runs[partition_id].get_hyperparams(partition_id)
            all_hyperparams = pd.concat([all_hyperparams, partition_hyperparams])
        return all_hyperparams

    def show_evaluation(self, metric: str = 'accuracy') -> pd.DataFrame:
        """
        Summarize the evaluation metric of choice across all partitions. Metric must be pre-calculated by ModelRun by
         ModelRun.evaluate_model().
        Args:
            metric: Metric to summarize from ModelRun.evaluation dataframe.

        Returns: Summary of the metric across all partitions, inlcuding mean, median, and stddev.
        """
        all_accuracy = {}
        for partition_id in self.model_runs:
            all_accuracy[partition_id] = self.model_runs[partition_id].evaluation[metric]
        all_accuracy_df = pd.DataFrame(all_accuracy, index=[self.name])
        median = all_accuracy_df.median(axis=1)
        mean = all_accuracy_df.mean(axis=1)
        stddev = all_accuracy_df.std(axis=1)
        all_accuracy_df['mean'] = mean
        all_accuracy_df['median'] = median
        all_accuracy_df['stddev'] = stddev
        return all_accuracy_df.sort_values('median', ascending=False)

    def generate_predictor(self, partition: int = 0) -> Predictor:
        """
        Return a Predictor from the trained model of a specific partition.

        Args:
            partition: The partition id of the model to return. Defaults to 0.

        Returns: a Predictor.
        """
        return self.model_runs[partition].generate_predictor()


def split_df_to_train_validate_test(df_input: pd.DataFrame, train_percent=0.7, validate_percent=0.15):
    """
    Args:
         df_input: dataframe with all variables of interest for the model
    Returns: dataframe with variables split into train, validation and test sets
    """

    df_output = df_input.copy()

    seed = 0
    np.random.seed(seed)
    perm = np.random.permutation(df_output.index)
    df_len = len(df_output.index)
    train_end = int(train_percent * df_len)
    validate_end = int(validate_percent * df_len) + train_end
    train = df_output.iloc[perm[:train_end]]
    validate = df_output.iloc[perm[train_end:validate_end]]
    test = df_output.iloc[perm[validate_end:]]

    return train, validate, test
