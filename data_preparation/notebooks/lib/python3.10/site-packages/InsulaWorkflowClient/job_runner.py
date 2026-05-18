import requests
from time import sleep
from .InsulaApiConfig import InsulaApiConfig
from .job_params import InsulaJobParams
from .job_status import InsulaJobStatus
from .logger import logger


class InsulaRunner(object):
    def __init__(self, insula_config: InsulaApiConfig):
        super().__init__()
        self.__insula_api_config = insula_config
        self.__platform_status_attempts = 0
        self.__job_status_attempts = {}

    def __set_job_config(self, params: InsulaJobParams):

        attempt = 0
        last_response = ''
        while attempt < 3:
            r = requests.post(
                self.__insula_api_config.job_config_api_path,
                data=str(params),
                headers=self.__insula_api_config.headers,
                verify=self.__insula_api_config.disable_ssl_check == False
            )

            if r.status_code != 201:
                attempt += 1
                last_response = r.text
                sleep(5)
            else:
                resp_dict = r.json()
                return resp_dict['id']

        raise Exception(f'Cant create Job_config: {last_response}')

    def __launch_job(self, config_id):
        attempt = 0
        last_response = ''
        while attempt < 3:
            url = self.__insula_api_config.get_job_launch_api_path(config_id)
            run_request = requests.post(url, headers=self.__insula_api_config.headers,
                                        verify=self.__insula_api_config.disable_ssl_check == False)
            if run_request.status_code != 202:
                attempt += 1
                last_response = run_request.text
                sleep(5)
            else:
                run_request_dict = run_request.json()
                return run_request_dict['id']

        raise Exception(f'Cant lunch the Job_config: {config_id}, error: {last_response}')

    def __add_attempt_to_job(self, job_id) -> int:
        str_job_id = str(job_id)

        if str_job_id not in self.__job_status_attempts:
            self.__job_status_attempts[str_job_id] = 0

        self.__job_status_attempts[str_job_id] += 1
        return self.__job_status_attempts[str_job_id]

    @staticmethod
    def __log_return_status(job_id, status):
        logger.info(f'job id: {job_id} status: {status}')
        return status

    def __get_job_status(self, job_id):
        # print('__get_job_status...')
        logger.info(f'Check status job id: {job_id}')
        url_status = self.__insula_api_config.get_job_status_api_path(job_id)
        status = requests.get(url_status, headers=self.__insula_api_config.headers,
                              verify=self.__insula_api_config.disable_ssl_check == False)

        if status.status_code != 200:
            if self.__platform_status_attempts >= 3:
                raise Exception(f'Cant get status job: {job_id}')
            else:
                logger.info(f'Cant get status job: {job_id} at attempt {self.__platform_status_attempts}')
                self.__platform_status_attempts += 1
                return self.__log_return_status(job_id, 'RUNNING')

        url_status_dict = status.json()

        # print(str(url_status_dict))

        job_status = url_status_dict['status']
        if job_status == 'COMPLETED' or job_status == 'CANCELLED':
            return self.__log_return_status(job_id, job_status)

        if job_status == 'ERROR':
            try:
                return self.__log_return_status(job_id, self.__check_error_and_retry(job_id, url_status_dict))
            except Exception as e:
                self.__platform_status_attempts += 1
                return self.__log_return_status(job_id, 'RUNNING')

        self.__platform_status_attempts = 0
        return self.__log_return_status(job_id, 'RUNNING')

    def __check_error_and_retry(self, job_id, url_status_dict):
        # print('__check_error_and_retry')
        if self.__is_parent_job(url_status_dict):
            # print('is parent')
            return self.__check_sub_jobs_and_retry(job_id)
        else:
            # print('single')
            if self.__add_attempt_to_job(job_id) > self.__insula_api_config.max_processor_attempts:
                logger.info(f'max retry{job_id}, retry: {self.__job_status_attempts[str(job_id)]}')
                return 'ERROR'
            else:
                # print('retry')
                self.__retry_job(job_id)
                return 'RUNNING'

    def __check_sub_jobs_and_retry(self, parent_job_id):
        sub_jobs_list_resp = requests.get(self.__insula_api_config.get_sub_jobs(parent_job_id),
                                          headers=self.__insula_api_config.headers,
                                          verify=self.__insula_api_config.disable_ssl_check == False)
        # print(f'__check_sub_jobs_and_retry sub_jobs_list_resp {sub_jobs_list_resp.status_code} {sub_jobs_list_resp.text}')
        if sub_jobs_list_resp.status_code != 200:
            raise RuntimeError('cant retrieve sub jobs')

        sub_jobs = sub_jobs_list_resp.json()

        if not '_embedded' in sub_jobs or not 'jobs' in sub_jobs['_embedded']:
            return 'ERROR'

        for sub_job in sub_jobs['_embedded']['jobs']:
            if sub_job['status'] == 'ERROR':
                sub_job_id = sub_job['id']
                if self.__add_attempt_to_job(sub_job_id) > self.__insula_api_config.max_processor_attempts:
                    logger.info(f'max retry{sub_job_id}, retry: {self.__job_status_attempts[str(sub_job_id)]}')
                    return 'ERROR'
                else:
                    self.__retry_job(sub_job_id)

        return 'RUNNING'

    def __is_parent_job(self, url_status_dict) -> bool:
        # print('__is_parent_job')
        res_config = requests.get(url_status_dict["_links"]["config"]["href"], headers=self.__insula_api_config.headers,
                                  verify=self.__insula_api_config.disable_ssl_check == False)
        if res_config.status_code != 200:
            raise RuntimeError('cant retrieve job config')

        job_config_x = res_config.json()
        return job_config_x['_embedded']["service"]["type"] == 'PARALLEL_PROCESSOR'

    def __retry_job(self, job_id):
        logger.info(f'Retry job: {job_id}...')
        status = requests.post(self.__insula_api_config.relaunch_failed_job(job_id),
                               headers=self.__insula_api_config.headers,
                               verify=self.__insula_api_config.disable_ssl_check == False)

        if status.status_code != 200:
            raise RuntimeError(f'Failed retry job: {job_id}')

        logger.info(f'Retry job: {job_id}... status_code: {status.status_code}')

        return status

        # TODO: To create run from config_id

    def run(self, job_params: InsulaJobParams, **kwargs) -> InsulaJobStatus:
        insula_job_status = InsulaJobStatus()
        insula_job_status.set_properties({'params': str(job_params), 'kwargs': kwargs})

        try:
            insula_job_status.set_config_id(self.__set_job_config(job_params)).save()
            sleep(self.__insula_api_config.interval_between_requests)
            insula_job_status.set_job_status('LAUNCHING').save()
            insula_job_status.set_job_id(self.__launch_job(insula_job_status.get_config_id())).save()
            insula_job_status.set_job_status('RUNNING')
            while insula_job_status.get_job_status() == 'RUNNING':
                sleep(self.__insula_api_config.status_polling_interval)
                insula_job_status.set_job_status(self.__get_job_status(insula_job_status.get_job_id())).save()

        except Exception as error:
            insula_job_status.set_job_error('ERROR', error).save()

        insula_job_status.remove_if_completed()
        return insula_job_status
