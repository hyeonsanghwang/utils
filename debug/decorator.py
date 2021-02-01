import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from time import perf_counter


# Decorator
def debug_time(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        ret = func(*args, **kwargs)
        end = perf_counter()
        print(f"Processing time (function '{func.__name__}'): {(end-start) * 1000:.03f} ms")
        return ret
    return wrapper


# Decorator
def debug_trace(func):
    def wrapper(*args, **kwargs):
        print(f"[Start] Processing (function '{func.__name__}')")
        ret = func(*args, **kwargs)
        print(f"[End] Processing (function '{func.__name__}')")
        return ret
    return wrapper
