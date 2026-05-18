from .singleton import SingletonMeta


class SnakesBus(metaclass=SingletonMeta):
    __subscribers = {}

    @classmethod
    def subscribe(cls, event_type, subscriber):
        if event_type not in cls.__subscribers:
            cls.__subscribers[event_type] = []
        cls.__subscribers[event_type].append(subscriber)

    @classmethod
    def publish(cls, event_type, data=None):
        if event_type in cls.__subscribers:
            for subscriber in cls.__subscribers[event_type]:
                subscriber.handle_event(event_type, data)
