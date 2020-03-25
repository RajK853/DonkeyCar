from time import time
from functools import wraps


def timer_wrapper(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        dt = time()-t0
        print(f"- Function '{func.__name__}' took {dt:.3f} seconds!\n")
        return result
    return wrapper_func


class ContextManagerWrapper:
    # A class wrapper to convert normal object into context managers
    def __init__(self, obj, exit_method):
        self.obj = obj
        self.exit_method = getattr(obj, exit_method)

    def __enter__(self):
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_method()