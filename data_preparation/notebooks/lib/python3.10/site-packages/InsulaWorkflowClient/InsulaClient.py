from .utils import InsulaUtils
from .InsulaApiConfig import InsulaApiConfig
from .workflow import InsulaWorkflow


class InsulaClient(object):
    def __init__(self, insula_config: InsulaApiConfig):
        self.__insula_api_config = insula_config

    def run_from_file(self, filename, parameters: dict, name=None):
        return self.run(InsulaUtils.load_from_file(filename), parameters, name)

    def run(self, content: str, parameters: dict, name=None):
        wf = InsulaWorkflow(self.__insula_api_config, content, parameters, name)
        return wf.run()
