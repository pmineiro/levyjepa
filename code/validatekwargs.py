from functools import wraps
from typing import Union, Callable

def validate_kwargs(**checks: dict[str, tuple[type, Callable]]):
    """
    Decorator to validate keyword arguments for *any* instance method.

    Example:
        @validate_kwargs(alpha=(float, lambda x: x > 0))
        def set_alpha(self, *, alpha): ...
    """
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            for name, (typ, cond) in checks.items():
                if name not in kwargs:
                    continue
                val = kwargs[name]

                # Handle `float | None` union types
                if getattr(typ, "__origin__", None) is Union:
                    if not any(isinstance(val, t) for t in typ.__args__):
                        raise TypeError(f"{name} must be one of {typ.__args__}")
                else:
                    if not isinstance(val, typ):
                        raise TypeError(f"{name} must be {typ}")

                if cond is not None and not cond(val):
                    raise ValueError(f"{name} failed constraint")
            return method(self, *args, **kwargs)
        return wrapper
    return decorator
