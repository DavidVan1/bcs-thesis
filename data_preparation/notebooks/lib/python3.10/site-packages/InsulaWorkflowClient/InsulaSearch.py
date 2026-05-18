import requests
from .InsulaApiConfig import InsulaApiConfig
from .InsulaQuery import InsulaQuery


class InsulaSearch(object):
    __page_for_page = 'page={p}&resultsPerPage={pp}'

    def __init__(self, insula_config: InsulaApiConfig, **kwargs):
        super().__init__()
        self.__insula_api_config = insula_config

        self.results_for_page = kwargs.get('results_for_page')
        if self.results_for_page is None:
            self.results_for_page = 100

    def query(self, query: InsulaQuery, call_back=None, **kwargs):
        template = f'{self.__insula_api_config.search_api_path}?{self.__page_for_page}&{str(query)}'
        for i in range(0, 10000):
            url = template.format(p=i, pp=self.results_for_page)
            start_data = self.__load_data_from_url(url)
            features = start_data['features']
            if len(features) == 0:
                break
            if call_back is not None:
                call_back(features, **kwargs)
            else:
                query.load(features, **kwargs)

    def __load_data_from_url(self, url_search_metadata) -> dict:
        search_result = requests.get(url_search_metadata, headers=self.__insula_api_config.headers, verify=self.__insula_api_config.disable_ssl_check==False)

        # TODO: return the error!!!!
        if search_result.status_code != 200:
            return {}

        search_result_dict = search_result.json()
        return search_result_dict
