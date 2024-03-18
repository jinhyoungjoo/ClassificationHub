from typing import Any, Union


class Registry(object):
    """Singleton pattern registry that holds all classes."""

    instance_ = None
    objects_ = {}

    def __new__(cls):
        if cls.instance_ is not None:
            return cls.instance_

        cls.instance_ = super(Registry, cls).__new__(cls)
        cls.objects_ = {}

    @classmethod
    def register(cls, prefix: Union[str, None] = None):
        """Decorator for object registration.

        Args:
            prefix (str): The prefix appended to the object name saved in the
              registry.

        """

        def decorator(object: Any):
            object_name = object.__name__

            if prefix is None:
                cls.objects_[object_name.lower()] = object
            else:
                cls.objects_[f"{prefix}_{object_name.lower()}"] = object

            return object

        return decorator

    @classmethod
    def get(cls, name: str, prefix: Union[str, None] = None) -> Any:
        """Retrieve the object registered in the registry.

        Args:
            name (str): The name of the object to retrieve.
            prefix (str): The prefix appended to the object name saved in the
              registry.

        Returns:
            Any: The object registered in the registry.

        """
        if prefix is None:
            name = name.lower()
        else:
            name = f"{prefix}_{name.lower()}"

        object = cls.objects_.get(name, None)
        if object is None:
            raise KeyError(f"No object named '{name}' found in registry!")

        return object

    def __repr__(self) -> str:
        items = [f"{key}: {object}" for key, object in self.objects_.items()]
        return "\n".join(items)
