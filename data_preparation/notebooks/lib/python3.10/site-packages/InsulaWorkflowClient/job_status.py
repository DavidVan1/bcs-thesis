from os import remove
from datetime import datetime
from .utils import InsulaUtils


class InsulaJobStatus(object):
    def __init__(self) -> None:
        super().__init__()
        self.__properties = {}
        # 'insula_run'
        # '/home/worker/workDir/outDir/output'
        self.__pid_path = 'insula_run'
        self.__job_id = 0
        self.__config_id = 0
        self.__status = ''
        self.__error = ''
        self.__create_date = self.__get_now()
        self.__last_update_date = self.__get_now()
        InsulaUtils.create_folder_if_not_exists(self.__pid_path)

    def __str__(self):
        return InsulaUtils.get_json_formatted(self.get_status())

    def get_status(self) -> dict:
        return {
            'config_id': self.__config_id,
            'job_id': self.__job_id,
            'status': self.__status,
            'error': self.__error,
            'properties': self.__properties,
            'create_date': self.__create_date,
            'last_update_date': self.__last_update_date
        }

    def update_status_last_update_date(self):
        self.__last_update_date = self.__get_now()

    def save(self):
        self.__last_update_date = self.__get_now()
        self.__write_in_pid()

    def load(self):
        if self.__job_id != 0:
            pid_loaded_from_file = InsulaUtils.create_dict_from_json_file(self.__get_pid_name())
            self.__config_id = pid_loaded_from_file['config_id']
            self.__job_id = pid_loaded_from_file['job_id']
            self.__status = pid_loaded_from_file['status']
            self.__error = pid_loaded_from_file['error']
            self.__properties = pid_loaded_from_file['properties']
            self.__create_date = pid_loaded_from_file['create_date']
            self.__last_update_date = pid_loaded_from_file['last_update_date']

    @staticmethod
    def __get_now():
        return datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def __write_in_pid(self):
        if self.get_job_id() != 0:
            with open(self.__get_pid_name(), 'w') as f:
                f.write(str(self))

    def set_job_error(self, status, error):
        self.__error = str(error)
        self.__status = status
        self.update_status_last_update_date()
        return self

    def set_job_status(self, status):
        self.__status = status
        return self

    def get_job_status(self):
        return self.__status

    def set_config_id(self, config_id):
        self.__config_id = config_id
        return self

    def set_job_id(self, job_id):
        self.__job_id = job_id
        return self

    def get_job_id(self):
        return self.__job_id

    def get_config_id(self):
        return self.__config_id

    def set_properties(self, properties: dict):
        self.__properties = properties
        return self

    def get_properties(self):
        return self.__properties

    def remove(self):
        remove(self.__get_pid_name())

    def remove_if_completed(self):
        if self.__status == 'COMPLETED':
            self.remove()

    def __get_pid_name(self):
        return f'{self.__pid_path}/{self.get_job_id()}.json'
