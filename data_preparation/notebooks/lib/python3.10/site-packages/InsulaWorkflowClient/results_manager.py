from abc import ABC, abstractmethod
from snakenest import Snake


class ResultManager(ABC):

    @abstractmethod
    def get_result_steps(self, identifier) -> list:
        pass

    @abstractmethod
    def add_result_step(self, identifier, step):
        pass

    @abstractmethod
    def delete(self, identifier):
        pass

    @abstractmethod
    def name(self):
        pass


@Snake(having={'${insulaclient.result_manager.snake:memory}': 'memory'})
class MemoryResultsManager(ResultManager):

    def delete(self, identifier):
        if identifier in self.__step_results:
            del self.__step_results[identifier]

    def name(self):
        return 'MemoryResultsManager'

    def __init__(self):
        super().__init__()
        self.__step_results = {}

    def get_result_steps(self, identifier) -> list:
        """
        Returns a COPY list of dictionaries with the result of the workflow steps
        :return: List of COPY dictionaries with the result of the workflow steps
        """
        if identifier in self.__step_results:
            return self.__step_results[identifier].copy()

        return []

    def add_result_step(self, identifier, step):

        if identifier not in self.__step_results:
            self.__step_results[identifier] = []

        self.__step_results[identifier].append(step)
