from abc import ABC
from typing import Dict, Type


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

    @classmethod
    def to_dict(cls, component) -> Dict:
        d = {
            'flavor': type(component).__name__,
            'config': component.config,
        }
        return d

    # TODO it's dumb that these two base methods are the same!
    #  But need subclass to be able to override from_dict with additional info that is added post-initializatsion,
    #  without changing configure(), and need to be able to override configure with extra args to pass to constructor
    #  without overriding to_dict()
    @classmethod
    def configure(cls, d: Dict, **kwargs) -> ConfigurableComponent:
        """
        Instantiate a component from a dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        for required_key in ['flavor', 'config']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(config=d['config'])
        return flavor_instance

    @classmethod
    def from_dict(cls, d: Dict) -> ConfigurableComponent:
        """
        Instantiate a component from a dictionary.

        Args:
            d: A dictionary with the keys 'flavor' describing the class name of the component to be insantiated, and
             key 'config' containting the object's config dictionary. d may also contain other keys, which must be added
             to the object by the subclass-ed method.

        Returns:

        """
        for required_key in ['flavor', 'config']:
            if required_key not in d:
                raise ValueError(f"Component dictionary must contain key '{required_key}'.")

        flavor_cls = cls.select_flavor(d['flavor'])
        flavor_instance = flavor_cls(config=d['config'])
        return flavor_instance

    @classmethod
    def select_flavor(cls, flavor: str) -> Type[ConfigurableComponent]:
        if flavor in cls.registered_flavors:
            return cls.registered_flavors[flavor]
        else:
            raise ValueError(f"{cls.__name__} '{flavor}' not recognized")
