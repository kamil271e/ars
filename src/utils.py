import time


def tictoc(func: callable) -> callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed: {end - start}")
        return result
    return wrapper
