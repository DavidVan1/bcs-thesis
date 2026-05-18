import requests
from .InsulaApiConfig import InsulaApiConfig


class InsulaJobLogs(object):
    def __init__(self, insula_config: InsulaApiConfig):
        super().__init__()
        self.__insula_api_config = insula_config

    def __get_job_logs(self, job_id):
        run_request = requests.get(self.__insula_api_config.get_job_logs_api_path(job_id),
                                   headers=self.__insula_api_config.headers,
                                   verify=self.__insula_api_config.disable_ssl_check==False)

        if run_request.status_code != 200:
            return None

        return run_request.json()

    @staticmethod
    def __clean_job_logs(job_logs):
        if job_logs is None:
            return {}

        log = []
        for job_log in job_logs:
            log.append(job_log['message'].replace('\n', ''))

        return log

    def get_logs(self, job_id):
        return {'logs': self.__clean_job_logs(self.__get_job_logs(job_id))}
