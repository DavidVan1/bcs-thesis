import json
from os import makedirs
from os.path import exists


class InsulaUtils:
    @staticmethod
    def create_folder_if_not_exists(folder_name):
        if not exists(folder_name):
            makedirs(folder_name)

    @staticmethod
    def get_json_formatted(json_object):
        return json.dumps(json_object, indent=2)

    @staticmethod
    def load_from_file(file_name):
        data = None
        with open(file_name) as my_file:
            data = my_file.read()
        return data

    @staticmethod
    def write_file(file_name, content):
        with open(file_name, 'w') as f:
            f.write(content)

    @staticmethod
    def write_file_byte(file_name, content):
        with open(file_name, 'wb') as f:
            f.write(content)

    @staticmethod
    def create_dict_from_json_file(file_nane):
        content_file = InsulaUtils.load_from_file(file_nane)
        return json.loads(content_file)
