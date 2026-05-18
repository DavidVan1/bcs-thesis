import requests
import os
from re import findall
from .InsulaApiConfig import InsulaApiConfig
from .files_job_result import InsulaFilesJobResult
from .utils import InsulaUtils


class InsulaDownloadJobResults(object):
    def __init__(self, insula_config: InsulaApiConfig):
        super().__init__()
        self.__insula_api_config = insula_config

    def __get_job_results_from_job_id(self, job_id):
        job_result = InsulaFilesJobResult(self.__insula_api_config)
        return job_result.get_result_from_job(job_id)

    def __download_job_results(self, results: list, output_folder):
        for result in results:
            self.download_file(result['download'], output_folder)

    def download_file(self, file_to_download: str, output_folder):

        status = {
            'success': False,
            'path': None,
            'message': None
        }

        run_request = requests.get(file_to_download,
                                   headers=self.__insula_api_config.authorization_header,
                                   verify=self.__insula_api_config.disable_ssl_check==False)

        if run_request.status_code != 200:
            status['message'] = f"cant download file: {file_to_download}, status: {run_request.status_code}"

        headers = run_request.headers['content-disposition']
        filename = findall("filename=(.+)", headers)[0]
        my_path = os.path.join(output_folder, filename).replace('"', '')

        try:
            InsulaUtils.write_file_byte(my_path, run_request.content)
        except Exception as e:
            status['message'] = e
            return status

        status['success'] = True
        status['path'] = my_path
        return status

    def download_from_job_id(self, job_id, output_folder):
        results = self.__get_job_results_from_job_id(job_id)
        self.__download_job_results(results, output_folder)

    def download_from_job_results(self, results, output_folder):
        self.__download_job_results(results, output_folder)
