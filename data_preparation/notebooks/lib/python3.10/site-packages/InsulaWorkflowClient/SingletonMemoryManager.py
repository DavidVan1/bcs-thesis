import uuid
from threading import Lock


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]


class StoreManager(metaclass=SingletonMeta):

    def create_new_memory_block(self):
        pass

    def get_block(self, step_id: str = None) -> dict:
        pass

    def persist(self, step_id: str = None):
        pass


class MemoryManager(StoreManager):
    def __init__(self) -> None:
        self.__body = {}
        self.__lock: Lock = Lock()

    def create_new_memory_block(self):
        uid = uuid.uuid1()
        while True:
            if uid not in self.__body:
                self.__body[str(uid)] = {}
                break
            uid = uuid.uuid1()

        return str(uid)

    def get_block(self, step_id: str = None) -> dict:

        if step_id is None and len(self.__body) == 1:
            with self.__lock:
                return self.__body[next(iter(self.__body))]

        if step_id in self.__body:
            with self.__lock:
                new_me = self.__body[step_id].copy()

            return new_me

        return {}

    def persist(self, step_id: str = None):
        pass
