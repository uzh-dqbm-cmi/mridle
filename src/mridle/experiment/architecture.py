from abc import abstractmethod

import skorch
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb
from .ConfigurableComponent import ConfigurableComponent, ComponentInterface
from typing import Any, Dict, List, Tuple, Type, Union


class Architecture(ConfigurableComponent):

    def __init__(self, config):
        super().__init__(config)

    @abstractmethod
    def fit(self, x, y):
        pass


class ArchitectureInterface(ComponentInterface):

    registered_flavors = {
        'RandomForestClassifier': RandomForestClassifier,  # TODO enable auto-loading from sklearn
        'LogisticRegression': LogisticRegression,
        'XGBClassifier': xgb.XGBClassifier,
        'Pipeline': Pipeline,
    }

    config_schema = {
        'flavor': {
            'type': 'string',
        },
        'config': {
            'type': 'dict',
            'default': dict(),
        },
        'name': {
            'type': 'string',
            'required': False,
        },
        'args': {
            'type': 'dict',
            'required': False,
        },
        'instantiate': {
            'type': 'boolean',
            'default': True,
        }
    }

    @classmethod
    def configure(cls, d: Dict, **kwargs) -> Union[ConfigurableComponent, Type[ConfigurableComponent]]:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])

        if d.get('instantiate', True) is False:
            return flavor_cls

        elif flavor_cls == Pipeline:
            flavor_instance = cls.configure_pipeline(d)
        elif flavor_cls == ColumnTransformer:
            flavor_instance = cls.configure_column_transformer(d)
        elif flavor_cls == skorch.NeuralNet:
            flavor_instance = cls.configure_skorch_neural_net(d)
        elif flavor_cls == FunctionTransformer:
            flavor_instance = cls.configure_function_transformer(d)
        else:
            flavor_instance = flavor_cls(**d['config'])
        return flavor_instance

    @classmethod
    def deserialize(cls, d: Dict) -> ConfigurableComponent:
        """
        Instantiate a component from a {'flavor: ..., 'config': {}} dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        kwargs = cls.additional_info_for_deserialization(d)
        if flavor_cls == Pipeline:
            flavor_instance = cls.configure_pipeline(d)
        elif flavor_cls == ColumnTransformer:
            flavor_instance = cls.configure_column_transformer(d)
        else:
            flavor_instance = flavor_cls(**d['config'], **kwargs)
        return flavor_instance

    @classmethod
    def serialize(cls, component) -> Dict:
        if isinstance(component, Tuple):
            if isinstance(component[1], ColumnTransformer):
                return cls.serialize_column_transformer(component)
            else:
                return cls.serialize_pipeline_step(component)
        elif isinstance(component, Pipeline):
            return cls.serialize_pipeline(component)
        elif isinstance(component, FunctionTransformer):
            return cls.serialize_function_transformer(component)
        elif isinstance(component, skorch.NeuralNet):
            return cls.serialize_skorch_neural_net(component)
        elif isinstance(component, BaseEstimator):
            return cls.serialize_sklearn_estimator(component)
        else:
            return super().serialize(component)

    @classmethod
    def serialize_sklearn_estimator(cls, estimator) -> Dict:
        params = estimator.get_params()
        params = {k: params[k] for k in params.keys() if type(params[k]) != type}
        d = {
            'flavor': cls.get_instance_import_path(estimator),
            'config': params,
        }
        return d

    @staticmethod
    def get_instance_import_path(f) -> Dict:
        return f.__module__ + '.' + type(f).__name__

    @staticmethod
    def get_func_or_class_import_path(f) -> Dict:
        return f.__module__ + '.' + f.__name__

    @staticmethod
    def serialize_pipeline_step(step_tuple: Tuple[str, Any]) -> Dict:
        name, estimator = step_tuple
        d = ArchitectureInterface.serialize(estimator)
        d['name'] = name
        # d = {
        #     'flavor': estimator.__module__ + '.' + type(estimator).__name__,
        #     'name': name,
        #     'config': estimator.get_params(),
        # }
        return d

    @staticmethod
    def serialize_column_transformer_step(step_tuple: Tuple[str, Any, List[str]]) -> Dict:
        name, estimator, columns = step_tuple
        d = ArchitectureInterface.serialize(estimator)
        d['name'] = name
        d['args'] = {'columns': columns}
        # d = {
        #     'flavor': estimator.__module__ + '.' + type(estimator).__name__,
        #     'name': name,
        #     'args': {'columns': columns},
        #     'config': estimator.get_params(),
        # }
        return d

    @staticmethod
    def configure_pipeline(d):
        step_list = []
        if 'steps' not in d['config']:
            raise ValueError(f"Pipeline config must contain entry for `steps`, found {list(d['config'].keys())}")
        for step in d['config']['steps']:
            step_obj = ArchitectureInterface.configure(step)
            step_name = step.get('name', step['flavor'])
            step_tuple = (step_name, step_obj)
            step_list.append(step_tuple)
        pipeline_obj = Pipeline(step_list)
        return pipeline_obj

    @staticmethod
    def serialize_pipeline(pipeline) -> Dict:
        steps = []
        for step in pipeline.steps:
            serialized_step = ArchitectureInterface.serialize(step)
            steps.append(serialized_step)
        d = {
            'flavor': 'sklearn.pipeline.Pipeline',
            'config': {
                'steps': steps,
            }
        }
        print(d)
        return d

    @staticmethod
    def serialize_column_transformer(step_tuple) -> Dict:
        print('inside serialize_column_transformer')
        name, column_transformer = step_tuple
        steps = []
        for step in column_transformer.transformers:
            serialized_step = ArchitectureInterface.serialize_column_transformer_step(step)
            steps.append(serialized_step)
        d = {
            'flavor': 'sklearn.compose.ColumnTransformer',
            'name': name,
            'config': {
                'steps': steps
            }
        }
        print(d)
        return d

    @staticmethod
    def configure_column_transformer(d):
        step_list = []
        if 'steps' not in d['config']:
            raise ValueError(f"Pipeline config must contain entry for `steps`, found {list(d['config'].keys())}")
        for step in d['config']['steps']:
            step_obj = ArchitectureInterface.configure(step)
            step_name = step.get('name', step['flavor'])
            step_args = step['args']['columns']
            step_tuple = (step_name, step_obj, step_args)
            step_list.append(step_tuple)
        return ColumnTransformer(step_list)

    @staticmethod
    def configure_skorch_neural_net(d):
        arg_objects = {}
        for arg_name in d['args']:
            arg_obj = ArchitectureInterface.configure(d['args'][arg_name])
            arg_objects[arg_name] = arg_obj
        return skorch.NeuralNet(**arg_objects, **d['config'])

    @classmethod
    def serialize_skorch_neural_net(cls, net: skorch.NeuralNet) -> Dict:
        params = net.get_params()
        exclude = ['module', 'criterion', 'optimizer', 'iterator_train', 'iterator_valid', 'dataset', 'train_split',
                   'callbacks', '_kwargs']
        filtered_params = {k: params[k] for k in params if '__' not in k and k not in exclude}
        if '_kwargs' in params:
            filtered_params = d = {**filtered_params, **params['_kwargs']}

        d = {
            'flavor': 'skorch.NeuralNet',
            'config': filtered_params,
            'args': {
                'module': {
                    'flavor': cls.get_instance_import_path(net.module),
                    'config': net.module.get_params(),
                },
                'criterion': {
                    'flavor': cls.get_func_or_class_import_path(net.criterion),
                    'instantiate': False,
                },
                'optimizer': {
                    'flavor': cls.get_func_or_class_import_path(net.optimizer),
                    'instantiate': False,
                }
            },
        }
        return d

    @staticmethod
    def configure_function_transformer(d):
        if 'function' not in d['args']:
            raise ValueError("FunctionTransformer must contain an entry for 'function' in 'args'.")
        func = ArchitectureInterface.configure(d['args']['function'])
        return FunctionTransformer(func, **d['config'])

    @classmethod
    def serialize_function_transformer(cls, func_transformer):
        params = func_transformer.get_params()
        del params['func']
        d = {
            'flavor': 'sklearn.preprocessing.FunctionTransformer',
            'args': {
                'function':
                    {
                        'flavor': cls.get_func_or_class_import_path(func_transformer.func),
                        'instantiate': False,
                    },
            },
            'config': params
        }
        return d
