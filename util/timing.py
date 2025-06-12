import time
import functools

def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        if duration > 1:
            print(f"'{func.__name__}' levou {duration:.2f}s.")
        return result
    return wrapper
