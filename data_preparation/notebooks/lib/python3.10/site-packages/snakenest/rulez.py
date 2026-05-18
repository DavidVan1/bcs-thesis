from re import findall
from os.path import exists
from yaml import full_load as load_yaml_file
from os import environ


# https://matthewpburruss.com/post/yaml/
# https://pyyaml.org/wiki/PyYAMLDocumentation
class SnakeRulez:
    def __init__(self, file_):
        self.__cache = {}
        self.__rulez = {}
        if exists(file_):
            with open(file_) as f:
                self.__rulez = load_yaml_file(f)

    def __cache_result(self, name, value):
        self.__cache[name] = value
        return value

    def __get(self, name):

        if name in self.__cache:
            return self.__cache[name]

        res_env = self.__get_from_env(name)
        if res_env:
            return self.__cache_result(name, res_env)

        val = name.split('.')
        _max = len(val)
        obj = self.__rulez
        for i in range(_max):
            if obj and val[i] in obj:
                if i == _max - 1:
                    result = obj[val[i]]
                    return self.__cache_result(name, result)
                else:
                    obj = obj[val[i]]
            else:
                break
        return ''

    @staticmethod
    def __get_from_env(name: str):

        to_find = str(name)

        if to_find in environ:
            return environ[to_find]

        return ''

    def get(self, item: str, default=None):
        search = item
        default_item = default

        matches = findall('\\${(.*?)}', str(item))
        has_match = len(matches) > 0

        if has_match > 0:
            split = matches[0].split(':')
            default_item = split[1] if len(split) > 1 else None
            search = split[0]

        value = self.__get(search)
        if value != '':
            return value

        if has_match and default_item is None:
            raise AttributeError(f'rule {matches[0]} no set')

        return default_item
