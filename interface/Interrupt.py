import functools

def Interrupt(stop_event):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def check_interrupt():
                if stop_event.is_set():
                    print(f"[{func.__name__}] Execução interrompida.")
                    raise InterruptedError("Execução interrompida externamente.")

            result = func(*args, **kwargs)

            if hasattr(result, '__iter__') and not isinstance(result, str):
                for item in result:
                    check_interrupt()
                    yield item
            else:
                check_interrupt()
                return result
        return wrapper
    return decorator
