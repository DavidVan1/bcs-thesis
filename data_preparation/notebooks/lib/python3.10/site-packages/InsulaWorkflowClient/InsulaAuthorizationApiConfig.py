from abc import ABC, abstractmethod


class InsulaAuthorizationApiConfig(ABC):
    @abstractmethod
    def get_authorization_header(self):
        pass