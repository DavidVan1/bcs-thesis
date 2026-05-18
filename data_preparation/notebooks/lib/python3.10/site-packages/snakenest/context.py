import inspect
import types
from .configuration import Configuration
from .singleton import SingletonMeta


class SnakeParameters:
    def __init__(self, **kwargs):
        self.__name = kwargs.get("name", None)
        self.__args: dict = kwargs.get("args", {})
        self.__having = kwargs.get("having", None)
        self.__properties = kwargs.get("properties", None)
        self.__if_missing = kwargs.get("if_missing", True)

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name):
        if self.__name is None:
            self.__name = name

    @property
    def args(self) -> dict:
        return self.__args

    @args.setter
    def args(self, args):
        self.__args = args

    @property
    def having(self) -> dict:
        return self.__having

    @having.setter
    def having(self, condition):
        self.__having = condition

    @property
    def if_missing(self):
        return self.__if_missing


class Nest(metaclass=SingletonMeta):
    __nest = {'byQualifier': {}, 'byType': {}, 'definitions': {}}

    @classmethod
    def initialize(cls):
        cls.__initialize_snakes()

    @classmethod
    def definition_snake(cls, bean_definition, parameters: SnakeParameters):
        cls.__nest['definitions'][parameters.name] = (bean_definition, parameters)

    @classmethod
    def inject(cls, name, the_type):
        inst = cls.__get_class_by_name(name)
        if inst:
            return inst

        inst_array = cls.__get_class_by_type(the_type)
        if inst_array and len(inst_array) > 1:
            raise Exception(f'too many instances of {str(the_type)}')
        elif inst_array:
            inst = inst_array[0]

        if not inst:
            by_definitions = cls.__nest['definitions'].get(name, None)
            if by_definitions:
                return cls.__create_from_name(name, by_definitions)

        return inst

    @classmethod
    def clear(cls):
        if cls.__nest is not None:
            cls.__nest.clear()
            del cls.__nest
            cls.__nest = None
            cls.__init()

    @classmethod
    def __initialize_snakes(cls):
        for snake_name, snake_value in cls.__nest['definitions'].items():
            if snake_name not in cls.__nest['byQualifier']:
                cls.__create_from_name(snake_name, snake_value)

    @classmethod
    def __create_from_name(cls, snake_name, snake_definition):
        snake_class = snake_definition[0]
        snake_args: SnakeParameters = snake_definition[1]

        # update value from config
        if snake_args:
            for name, value in snake_args.args.items():
                snake_args.args[name] = NestConfiguration.get(value, value)

        if type(snake_class) is not types.FunctionType:
            snake_inst = cls.__create_from_class(snake_class, snake_args)
            if snake_inst:
                cls.__register_by_name(snake_name, snake_inst)
                cls.__register_by_type(snake_inst)

    @classmethod
    def __create_from_class(cls, snake_class, snake_args: SnakeParameters):

        if not Nest.__checks_snake_parameters(snake_args):
            return None

        tmp_args = []
        args_specs = inspect.getfullargspec(snake_class.__init__)[0]
        for i in range(len(args_specs)):

            if args_specs[i] in ['self', 'cls']:
                continue

            tmp_args.append(snake_args.args.get(args_specs[i], None))

        return snake_class(*tmp_args)

    @staticmethod
    def __checks_snake_parameters(snake_args: SnakeParameters):
        if snake_args.having:
            for keyme, value in snake_args.having.items():
                if NestConfiguration.get(keyme) != value:
                    return False

        return True

    @classmethod
    def __register_by_name(cls, name, inst):
        cls.__nest['byQualifier'][name] = inst

    @classmethod
    def __register_by_type(cls, inst):
        type_list = inspect.getmro(type(inst))
        for type_name in type_list:
            if type_name is type(object()):
                continue

            if cls.__nest['byType'].get(type_name, None) is None:
                cls.__nest['byType'][type_name] = []

            cls.__nest['byType'][type_name].append(inst)

    @classmethod
    def __init(cls):
        if cls.__nest is None:
            cls.__nest = {'byQualifier': {}, 'byType': {}, 'definitions': {}}

    @classmethod
    def __get_class_by_name(cls, name):
        return cls.__nest['byQualifier'].get(name, None)

    @classmethod
    def __get_class_by_type(cls, the_type):
        return cls.__nest['byType'].get(the_type, None)


NestConfiguration = Configuration()
