class StepResult(dict):

    # https://stackoverflow.com/questions/43836585/python-how-to-create-method-of-class-in-runtime
    def __init__(self, **kwargs):
        super().__init__()
        self.only_key = ['id', 'output_id', 'default', 'download', 'type', 'value']

        for key in self.only_key:
            if key in kwargs:
                self[key] = kwargs[key]

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if key in self.only_key:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        pass

    # def __call__(self, *args, **kwargs):
    #     pass

    def get(self, __key):
        return self[__key]
