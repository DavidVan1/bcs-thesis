class InsulaQuery(object):
    def __init__(self, params: dict, **kwargs):
        self.__feature = []
        self.__query_string = params

    def add_parameter(self, key, value):
        self.__query_string[key] = value

    def load(self, feature, **kwargs):
        self.__feature.extend(feature)

    def get_features(self):
        return self.__feature

    @staticmethod
    def __dict_to_query(d: dict):
        query = ''
        for key in d.keys():
            query += str(key) + '=' + str(d[key]) + "&"
        return query[:-1]

    def __str__(self):
        return self.__dict_to_query(self.__query_string)
