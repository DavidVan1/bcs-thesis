import uuid


class WorkflowData(object):
    def __init__(self, workflow_id: str = None):
        super().__init__()
        self.__uuid = workflow_id if workflow_id else str(uuid.uuid1())
        self.__name = 'UnName'
        self.__type = None
        self.__version = None
        self.__parameters = {}

        self.__requirements = {
            'connections': {},
            'jobs': {}
        }
        self.__config = {}
        self.__templates = {}
        self.__steps = []

    @property
    def identifier(self):
        return self.__uuid

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        self.__type = value

    @property
    def version(self):
        return self.__version

    @version.setter
    def version(self, value):
        self.__version = value

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, value):
        self.__parameters = value

    @property
    def requirements(self):
        return self.__requirements

    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self, config):
        self.__config = config

    @property
    def templates(self):
        return self.__templates

    @templates.setter
    def templates(self, value):
        self.__templates = value

    @property
    def steps(self):
        return self.__steps

    @steps.setter
    def steps(self, value):
        self.__steps = value
