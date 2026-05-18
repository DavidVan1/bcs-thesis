import inspect
from abc import ABC, abstractmethod
from .context import Nest, NestConfiguration, SnakeParameters
from .event import SnakesBus
from .rulez import SnakeRulez


class Snake:
    def __init__(self, **kwargs):
        self.__snake_parameters = SnakeParameters(**kwargs)

    def __call__(self, class_definition):
        self.__snake_parameters.name = class_definition.__name__
        Nest.definition_snake(class_definition, self.__snake_parameters)
        return class_definition


class Poisoned:

    def __init__(self, **kwargs):
        self.__config = kwargs

    def __call__(self, snake_method):
        def wrapper(*args, **kwargs):
            specs = inspect.getfullargspec(snake_method)
            args_list = specs[0]
            annotations = specs[6]
            real_args = []
            for i in range(len(args_list)):
                arg = args_list[i]
                real_args.append(
                    args[i] if len(args) > i and args[i] is not None
                    else Nest.inject(self.__config.get(arg, arg), annotations.get(arg))
                )

            return snake_method(*real_args, **kwargs)

        return wrapper


class SnakeNotificationSubscriber(ABC):
    @abstractmethod
    def handle_event(self, event_type, data=None):
        pass


__all__ = ['Nest', 'NestConfiguration', 'Snake',
           'Poisoned', 'SnakesBus', 'SnakeNotificationSubscriber',
           'SnakeRulez']
