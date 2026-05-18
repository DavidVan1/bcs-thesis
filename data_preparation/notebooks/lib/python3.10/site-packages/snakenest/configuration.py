from .rulez import SnakeRulez
from os import environ
from .singleton import SingletonMeta


class Configuration(metaclass=SingletonMeta):

    def __init__(self):
        # print(environ)
        self.__rulez = SnakeRulez(environ.get('NEST_HOME', 'nest.yaml'))

    def get(self, key: str, default=None):
        return self.__rulez.get(key, default)
