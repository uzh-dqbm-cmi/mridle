# Experiment
Name ideas:
* blocks
* ml-blocks
* mlblox
* exblox
* exblocks
* experiment-blocks
* stack
* skstack (taken)
* tower
---
# Approach
/Experiment/ turns machine learning model experimentation into a plug-and-play experience.
Tired of writing the same machine learning boilerplate code over and over? Feel like making a small change to your modeling experiment should be easy, and not involve re-running dozens of lines of code?

/Experiment/ is designed around the idea that a modeling experiment has three main parts: the data, the training, and the results.

Likewise, an Experiment has 3 parts:
1. A `DataSet`, which not only holds the data but also is responsible for keeping track of the features, target, and train/test partitions.
2. A `Trainer` that controls how the model is trained. `Trainer` holds an `Architecture`, which defines the model architecture to be trained. A `Trainer` may optionally also have a `Tuner`, which governs hyperparameter search.
3. The list of `Metrics` that the model(s) is evaluated on.

An `Experiment` may optionally also have a `metadata` dictionary that can store whatever information you want to track.

![Experiment Class Diagram](./ExperimentClassDiagram.svg)

# Example usage

## Define an `Experiment`
There are 2 ways to define an experiment: via object creation or via a config. 
If you are working in a notebook, you may want to use the object-creation method, and plug-and-play with different experiment components.
Once you have decided on a model definition you are happy with, you might find it helpful to write it as a configuration for readability and re-use. This approach has the advantage of providing a single point of truth for everything that contributes to the experiment as an easy-to-read configuration. 

### Create an `Experiment` from objects

```python
from dataset import DataSet
from stratifier import TrainTestStratifier
from sklearn.ensemble import RandomForestClassifier
from trainer import Trainer
from tuner import RandomSearchTuner
from metric import F1_Macro, AUPRC
from experiment import Experiment

dataset_config = {
    'features': ['A', 'B', 'C', 'D'],
    'target': 'E',
}
dataset = DataSet(dataset_config, df)
stratifier = TrainTestStratifier({'test_split_size': 0.3})
architecture = RandomForestClassifier()
tuner_config = {
    'hyperparameters': {
        'n_estimators': range(200, 2000, 10),
        'max_features': ['auto', 'sqrt'],
        'max_depth': range(10, 110, 11),
        'min_samples_split': [2, 4, 6, 8, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'bootstrap': [True, False],
    },
    'num_cv_folds': 3,
    'num_iters': 5,
    'scoring_function': 'f1_macro',
    'verbose': 1,
}
tuner = RandomSearchTuner(tuner_config)
trainer = Trainer(architecture=architecture, tuner=tuner)
metrics = [F1_Macro, AUPRC]
metadata = {'name': 'random forest'}

exp = Experiment(dataset, stratifier, trainer, metrics, metadata)
```

### Create an `Experiment` from configuration
When defining an Experiment via a config, all components and their configurations are written out in the structure below.
The only object that is not configured here is the dataframe- that is passed to Experiment separately.

```python
from experiment import Experiment

config = {
        'DataSet': {
            'flavor': 'DataSet',
            'config': {
                'features': ['A', 'B', 'C', 'D'],
                'target': 'E',
            },
        },
        'Stratifier': {
            'flavor': 'TrainTestStratifier',
            'config': {
                'test_split_size': 0.3,
            },
        },
        'Architecture': {
            'flavor': 'RandomForestClassifier',
            'config': {}
        },
        'Trainer': {
            'flavor': 'Trainer',
            'config': {}
        },
        'Tuner': {
            'flavor': 'RandomSearchTuner',
            'config': {
                'hyperparameters': {
                    'n_estimators': range(200, 2000, 10),
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': range(10, 110, 11),
                    'min_samples_split': [2, 4, 6, 8, 10],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'bootstrap': [True, False],
                },
                'num_cv_folds': 3,
                'num_iters': 5,
                'scoring_function': 'f1_macro',
                'verbose': 1,
            },
        },
        'Metrics': [
            {'flavor': 'F1_Macro', 'config': {'classification_cutoff': 0.5}},
            {'flavor': 'AUPRC', 'config': {}},
            {'flavor': 'AUROC', 'config': {}},
            {'flavor': 'LogLoss', 'config': {}},
        ]
    }

exp = Experiment.configure(config, df)
```

## Running an `Experiment`
To run an experiment, just call `experiment.go()`.

After a completed `go()` is complete, `Experiment will contain the following data:
* 'Experiment.evaluation': a dataframe with the evaluations of each partition's model on the specified `metrics`.
* 'Experiment.partition_predictors': The predictors that were trained during each experiment.
* 'Experiment.final_predictor': The predictor that was trained on the full dataset.
* 'Experiment.partition_training_metadata': Metadata collected during the training processes of each partition.
* 'Experiment.final_training_metadata': Metadata collected during the training processes of the `final_predictor`.

## Saving and Loading Experiments
### Save
Before saving the Experiment to a file, it should be serialized. This exports the experiment definition to (mostly) a configuration dictionary. This reduces the risk of being unable to re-load the experiment due to changes in the `Experiment` infrastructure.

```python
serialized_exp = exp.serialize()
with open(experiment_file_path, "wb+") as f:  # TODO: check syntax
    pickle.dump(serialized_exp, f)

```

### Load
After loading a serialized experiment from a pickle, deserialize it:
```python
with open(experiment_file_path, "rb+") as f:
    serialized_exp = pickle.load(f)
exp = Experiment.deserialize(serialized_exp)
```

## Using an `Experiment` to make novel predictions

```python
# create a new `DataSet` object from your new features_df using the experiment's dataset configuration.
data_set = DataSet(exp.dataset.config, features_df)
preds_proba = exp.final_predictor.predict_proba(data_set.x)
```
