from typing import Any

def checker(func: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any):
        for kwarg in kwargs:
            if kwargs[kwarg] == getattr(self, kwarg):
                return None
            else:
                setattr(self, kwarg, kwargs[kwarg])
        return func(self, *args, **kwargs)
    return wrapper