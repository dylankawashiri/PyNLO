from typing import Any

def checker(func: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any):
        if all(kwargs[kwarg] == getattr(self, kwarg, None) for kwarg in kwargs):
            return None
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])
        return func(self, *args, **kwargs)
    return wrapper