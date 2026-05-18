import requests
from time import sleep
from .InsulaApiConfig import InsulaApiConfig
from .step_result import StepResult


class InsulaDeleter(object):

    def __init__(self, insula_config: InsulaApiConfig):
        super().__init__()
        self.__insula_api_config = insula_config

    def delete_files(self, step_results: list[StepResult]):
        for step_result in step_results:
            self.delete_file(step_result)

    def delete_file(self, step_result: StepResult):
        file_to_delete = self.__insula_api_config.get_delete_platform_file(step_result.get('id'))
        run_request = requests.delete(file_to_delete,
                                      headers=self.__insula_api_config.authorization_header,
                                      verify=self.__insula_api_config.disable_ssl_check==False)
        # logger.info(f'deleted file: {file_to_delete} status code: {run_request.status_code}')
        sleep(self.__insula_api_config.delete_interval)
