from abc import ABC
from cerberus import Validator
from typing import Any, Dict, Type


class ConfigurableComponent(ABC):

    def __init__(self, config: Dict = None):
        self.config = config
        if self.config is None:
            self.config = {}

    # TODO - how to allow for different args in subclasses?
    # @classmethod
    # def validate_config(cls, config: Dict) -> bool:
    #     return True


class ComponentInterface:

    registered_flavors = {}

    config_schema = {
        'flavor': {
            'type': 'string',
        },
        'config': {
            'type': 'dict',
            'default': dict(),
        }
    }
    serialization_schema = {}

    @classmethod
    def serialize(cls, component) -> Dict:
        d = {
            'flavor': type(component).__name__,
            'config': component.config,
        }
        more_info = cls.additional_info_for_serialization(component)
        for key in more_info:
            d[key] = more_info[key]

        cls.validate_serialization_config(d)
        return d

    @classmethod
    def additional_info_for_serialization(cls, component: ConfigurableComponent) -> Dict[str, Any]:
        """
        (Optional) Build a dictionary of additional data (beyonnd the config) that needs to be saved in the
         serialization dictionary. The key value pairs will be added to the object's serialization dictionary.

        Args:
            component: The component from which to serialize additional information.

        Returns: A dictionary of additional information to be included in the serialization dictionary.
        """
        return {}

    # TODO it's dumb that these two base methods are the same!
    #  But need subclass to be able to override deserialize with additional info that is added post-initializatsion,
    #  without changing configure(), and need to be able to override configure with extra args to pass to constructor
    #  without overriding serialize()
    @classmethod
    def configure(cls, d: Dict, **kwargs) -> ConfigurableComponent:
        """
        Instantiate a component from a dictionary and other objects if necessary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(config=d['config'], **kwargs)
        return flavor_instance

    @classmethod
    def deserialize(cls, d: Dict) -> ConfigurableComponent:
        """
        Instantiate a component from a dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        d = cls.validate_serialization_config(d)
        flavor_cls = cls.select_flavor(d['flavor'])
        kwargs = cls.additional_info_for_deserialization(d)
        flavor_instance = flavor_cls(config=d['config'], **kwargs)
        return flavor_instance

    @classmethod
    def additional_info_for_deserialization(cls, d: Dict) -> Dict[str, Any]:
        """
        (Optional) Build a dictionary of additional data (beyond the config) that needs to be extracted from the
         serialization dictionary in order to initialize the component. The key value pairs will be passed to the
          component initialization as kwargs.

        Args:
            d: The serialization dictionary from which to extract additional information for initalization.

        Returns: A dictionary of additional information to be included passed to the component's init.
        """
        return {}

    @classmethod
    def select_flavor(cls, flavor: str) -> Type[ConfigurableComponent]:
        if flavor in cls.registered_flavors:
            return cls.registered_flavors[flavor]
        else:
            raise ValueError(f"{cls.__name__} '{flavor}' not recognized")

    @classmethod
    def validate_config(cls, d: Dict) -> Dict:
        v = Validator(cls.config_schema)
        d_norm = v.normalized(d)
        if not v.validate(d_norm):
            raise ValueError(f"{cls.__name__} encountered config validation errors: {v.errors}")
        return d_norm

    @classmethod
    def validate_serialization_config(cls, d: Dict) -> Dict:
        # compile config and serialization schemas
        schema = {}
        for key in cls.config_schema:
            schema[key] = cls.config_schema[key]
        for key in cls.serialization_schema:
            schema[key] = cls.serialization_schema[key]

        v = Validator(schema)
        d_norm = v.normalized(d)
        if not v.validate(d_norm):
            raise ValueError(f"{cls.__name__} encountered config validation errors: {v.errors}")
        return d_norm
