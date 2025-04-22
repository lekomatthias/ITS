import time
import functools

# Decorador para contar o tempo
def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"Função '{func.__name__}' demorou: {exec_time:.4f} segundos.")
        return result
    return wrapper
