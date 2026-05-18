import requests
from .InsulaApiConfig import InsulaApiConfig
from .job_status import InsulaJobStatus
from .step_result import StepResult
from time import sleep


class InsulaFilesJobResult(object):
    __page_for_page = 'page={p}&size={pp}'

    def __init__(self, insula_config: InsulaApiConfig):
        super().__init__()
        self.__insula_api_config = insula_config

    @staticmethod
    def __get_files_from_result_job(raw_results: dict):
        results = []
        for platform_file in raw_results['_embedded']['platformFiles']:
            results.append(StepResult(
                id=platform_file['id'],
                output_id=platform_file['filename'].split('/')[1],
                default=platform_file['uri'],
                download=platform_file["_links"]["download"]['href'],
                type='job_result'
            ))

        return results

    def get_result_from_job(self, job_id) -> list:
        max_loop = 10000
        per_page = 50
        raw_results = {
            '_embedded': {
                'platformFiles': []
            }
        }
        # raw_results['_embedded']['platformFiles'] = {}
        res = self.__insula_api_config.get_platform_files_search_parametric_find(job_id)
        template = f'{res}&{self.__page_for_page}'
        for i in range(0, max_loop):
            url = template.format(p=i, pp=per_page)

            run_request = requests.get(url,
                                       headers=self.__insula_api_config.headers,
                                       verify=self.__insula_api_config.disable_ssl_check == False)

            if run_request.status_code != 200:
                raise Exception(
                    f'cant retrieve result from job: {job_id}, status: {run_request.status_code}, text: {run_request.text}')

            raw = run_request.json()
            res_per_page = len(raw['_embedded']['platformFiles'])
            if res_per_page == 0:
                break

            raw_results['_embedded']['platformFiles'].extend(raw['_embedded']['platformFiles'])
            if res_per_page < per_page:
                break
            sleep(0.50)
            if i == (max_loop - 1):
                raise RuntimeError(f'Too many pages {i}')

        return self.__get_files_from_result_job(raw_results)

    def get_result_from_job_status(self, job_status: InsulaJobStatus) -> list:
        return self.get_result_from_job(job_status.get_job_id())

# https://insula.earth/secure/api/v2.0/platformFiles/search/parametricFind?page=0&size=20&sort=filename,asc&job=https:%2F%2Finsula.earth%2Fsecure%2Fapi%2Fv2.0%2Fjobs%2F10579&projection=detailedPlatformFile
