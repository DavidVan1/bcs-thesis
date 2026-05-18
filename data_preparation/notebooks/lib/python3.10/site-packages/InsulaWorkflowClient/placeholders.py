from re import findall
from snakenest import Poisoned
from .step_result import StepResult
from .WorkflowDataManager import WorkflowData
from .results_manager import ResultManager


class LinePlaceholder(object):
    __pattern = '\\${(.*?)}'

    def __init__(self, line):
        super().__init__()
        self.__line = line
        self.__all_match = findall(self.__pattern, line)
        self.__has_match = len(self.__all_match) > 0

        self.__match_dot_split = []
        if self.__has_match:
            self.__match_dot_split = self.__all_match[0].split('.')

    @property
    def line(self):
        return self.__line

    @property
    def has_matches(self) -> bool:
        return self.__has_match

    def get_matches(self) -> list:
        return self.__match_dot_split


class ResultFilterOnNamePlaceholder:
    __result_filter_pattern = '\\$\\[(.*?)]'

    def __init__(self, raw: str):
        self.__filters = findall(self.__result_filter_pattern, raw)

    def has_filters(self):
        return len(self.__filters) > 0

    def get_filters(self):
        return self.__filters

    def filter(self, filename):
        if self.has_filters():
            for filter_in in self.__filters:
                res = findall(filter_in, filename)
                if len(res) > 0:
                    return True
        return False


class ResultPlaceholder(object):

    @staticmethod
    def get_from_results(ipf: LinePlaceholder, raw, global_results: list) -> list[StepResult]:
        values = []
        matches = ipf.get_matches()
        step_output = None
        if len(matches) == 4:
            step_output = matches[3]

        ipf_filters = ResultFilterOnNamePlaceholder(raw)
        for step_results in global_results:
            if matches[2] == step_results['name']:
                for step_result in step_results['results']:
                    if step_output and step_result.get('output_id') != step_output:
                        continue

                    if ipf_filters.has_filters():
                        if ipf_filters.filter(step_result.get('default')):
                            values.append(step_result)
                    else:
                        values.append(step_result)
        return values


# -----------------------------------------
class Runtime(object):
    @staticmethod
    def get_runtime_info(lp: LinePlaceholder, global_parameters: list):

        step_id = lp.get_matches()[2]
        for step_results in global_parameters:
            if step_id == step_results['name']:
                out = lp.get_matches()[3]
                if out is not None and out in step_results['status']:
                    return [StepResult(default=step_results['status'][out], output_id=out, type='status')]

        return []
# -----------------------------------------

class ParameterPlaceholder(object):

    @staticmethod
    def placeholders(key: str, wd: WorkflowData) -> list[StepResult]:
        params = wd.parameters.get(key, None)
        if not params:
            return list()

        res = []
        for param in [params] if isinstance(params, str) else params:
            res.append(StepResult(default=param, value=param, type='param'))

        return res


class Placeholder(object):
    def __init__(self, line: str, data: WorkflowData):
        self.__results = self.__placeholders(line, data)
        super().__init__()

    def get_results(self):
        return self.__results

    def get_result(self) -> str:
        if len(self.__results) > 0:
            return self.__results[0].get('default')
        return ''

    # @staticmethod
    @Poisoned()
    def __placeholders(self, line: str, workflow_data: WorkflowData, result_manager: ResultManager) -> list[
        StepResult]:
        lp = LinePlaceholder(line)

        if not lp.has_matches:
            return [StepResult(default=line, value=line, type='placeholder')]

        match = lp.get_matches()
        if len(match) <= 2:
            return []

        if match[0] == 'workflow' and match[1] == 'step':
            return ResultPlaceholder.get_from_results(lp, line,
                                                      result_manager.get_result_steps(workflow_data.identifier))

        elif match[0] == 'workflow' and match[1] == 'param':
            if len(match) != 3:
                return []
            return ParameterPlaceholder.placeholders(match[2], workflow_data)

        elif match[0] == 'status' and match[1] == 'step':
            return Runtime.get_runtime_info(lp, result_manager.get_result_steps(workflow_data.identifier))

        return []
