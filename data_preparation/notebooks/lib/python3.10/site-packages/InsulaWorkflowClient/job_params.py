from .utils import InsulaUtils


class InsulaJobParams(object):
    def __init__(self, platform_uri_service_id):
        self.service_id = platform_uri_service_id
        self.params = {
            # 'service': f'https://insula.earth/secure/api/v2.0/services/{self.service_id}',
            'service': platform_uri_service_id,
            'inputs': {}
        }

    def set_inputs(self, input_name, input_value: list):
        if input_name not in self.params['inputs'].keys():
            self.params['inputs'][input_name] = []

        if len(input_value) > 0:
            # self.params['inputs'][input_name] = [','.join(input_value)]
            self.params['inputs'][input_name] = input_value
        else:
            self.params['inputs'][input_name] = []

    def __str__(self):
        return InsulaUtils.get_json_formatted(self.params)
