class InsulaWorkflowStep(object):
    def __init__(self, step_definitions):
        self.__step_definitions = []
        for step_definition in step_definitions:
            self.__step_definitions.append(step_definition)

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.__step_definitions):
            raise StopIteration

        step_definition = self.__step_definitions[self.__index]
        self.__index += 1
        return step_definition

    def count(self) -> int:
        return len(self.__step_definitions)

    def get_step(self, index):
        return self.__step_definitions[index]

    def get_steps(self):
        return self.__step_definitions

    def __str__(self):
        return f'Step: {self.__step_definitions}'
