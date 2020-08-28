import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from typing import Any, Dict, List, Tuple, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


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
        features = ['historic_no_show_cnt', 'no_show_before']
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


class PartitionedExperiment:

    def __init__(self, name: str, data_set: Any, label_key: str, preprocessing_func: Callable,
                 model_run_class: ModelRun, model, hyperparams: Dict, n_partitions: int = 5,
                 stratify_by_label: bool = True, feature_subset=None, reduce_features=False, verbose=False):
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

    def run(self, num_partitions_to_run=None, run_hyperparam_search=True):
        """
        Run the experiment.

        Args:
            num_partitions_to_run: select a subset of partitions to run, for faster testing.
            run_hyperparam_search: argument to turn off hyperparam search, for faster testing.

        Returns:

        """
        partitions_to_run = list(self.partition_ids.keys())
        if num_partitions_to_run is not None:
            partitions_to_run = partitions_to_run[:num_partitions_to_run]
            print("Running only partitions {}".format(", ".join(partitions_to_run)))

        for partition_name in self.partition_ids:
            if partition_name in partitions_to_run:
                print("Running partition {}...".format(partition_name))
                model_run = self.run_experiment_on_one_partition(data_set=self.data_set,
                                                                 label_key=self.label_key,
                                                                 partition_ids=self.partition_ids[partition_name],
                                                                 preprocessing_func=self.preprocessing_func,
                                                                 model_run_class=self.model_run_class,
                                                                 model=self.model,
                                                                 hyperparams=self.hyperparams,
                                                                 run_hyperparam_search=run_hyperparam_search,
                                                                 feature_subset=self.feature_subset,
                                                                 reduce_features=self.reduce_features)
                self.model_runs[partition_name] = model_run

        print("Compiling results")
        self.experiment_results = self.summarize_runs(self.model_runs)
        return self.experiment_results

    @classmethod
    def run_experiment_on_one_partition(cls, data_set: Dict, label_key: str, partition_ids: List[int],
                                        preprocessing_func: Callable, model_run_class: ModelRun, model,
                                        hyperparams: Dict, run_hyperparam_search: bool, feature_subset: List[str],
                                        reduce_features: bool):
        train_set, test_set = cls.materialize_partition(partition_ids, data_set)
        mr = model_run_class(train_set=train_set, test_set=test_set, label_key=label_key, model=model,
                             preprocessing_func=preprocessing_func, hyperparams=hyperparams,
                             feature_subset=feature_subset, reduce_features=reduce_features)
        mr.run(run_hyperparam_search=run_hyperparam_search)
        return mr

    @classmethod
    def partition_data_randomly(cls, data_set: pd.DataFrame, n_partitions: int) -> Dict[str, List[int]]:
        """Randomly shuffle and split the data set into n roughly equal partitions."""
        raise NotImplementedError

    @classmethod
    def partition_data_stratified(cls, label_list: List[int], n_partitions: int) -> Dict[str, List[int]]:
        """Randomly shuffle and split the doc_list into n roughly equal lists, stratified by label."""
        skf = StratifiedKFold(n_splits=n_partitions, random_state=42, shuffle=True)
        x = np.zeros(len(label_list))  # split takes a X argument for backwards compatibility and is not used
        partition_indexes = [test_index for train_index, test_index in skf.split(x, label_list)]
        partitions = {}
        for p_id, p in enumerate(partition_indexes):
            partitions['Partition {}'.format(p_id)] = partition_indexes
        return partitions

    @classmethod
    def materialize_partition(cls, partition_ids: List[int], data_set: pd.DataFrame)\
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training and testing dataset based on the partition, which indicate the ids for the test set.

        Args:
            partition_ids: Dictionary, where the values contain the indices of each partition's test set.
            data_set: The full data set.

        Returns: Train set and Test set.
        """

        train_set = data_set[~partition_ids]
        test_set = data_set[partition_ids]
        return train_set, test_set

    @classmethod
    def report_partition_stats(cls, partition_ids: Dict[str, List[int]], data_set: Any, label_key=str):
        """
        Print the size and class balance of each partition's train and test set.

        Args:
            partition_ids: Partitions.
            data_set: The full data set.
            label_key: The column name in data_set that is the label.

        Returns: None, just prints.
        """
        for partition_name in partition_ids:
            partition_ids = partition_ids[partition_name]
            train_set, test_set = cls.materialize_partition(partition_ids, data_set)
            labels_train = train_set[label_key]
            labels_test = test_set[label_key]

            print('\n-Partition {}-'.format(partition_name))
            print("Train: {:,.0f} data points".format(labels_train.shape[0]))
            print("Test: {:,.0f} data points".format(labels_test.shape[0]))

            label_options = data_set[label_key].nunique()
            for label_i in label_options:
                for data_subset_name, data_subset in {'Train': labels_train, 'Test': labels_test}:
                    pct_label_i = data_subset[data_subset[0] == label_i].shape[0] / data_subset.shape[0]
                    print("{} Set: {:.0%} {}".format(data_subset_name, pct_label_i, label_i))

    @classmethod
    def summarize_runs(cls, run_results: Dict):
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
