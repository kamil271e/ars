import time


def timer(func: callable) -> callable:
    """
    Timer decorator. Prints time elapsed during function execution.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{str(func)}] Time elapsed: {round(end - start, 4)} [s]")
        return result

    return wrapper
