"""Small helper for measuring elapsed time, so latency measurement doesn't
turn into scattered, easy-to-mismatch time.perf_counter() pairs across the
codebase.
"""
import time
from contextlib import contextmanager


class Elapsed:
    elapsed_seconds: float = 0.0


@contextmanager
def measure():
    """Usage:
        with measure() as timer:
            do_work()
        print(timer.elapsed_seconds)

    `timer.elapsed_seconds` is only meaningful after the `with` block
    exits -- it's set in a `finally`, so it's still recorded even if the
    wrapped code raises.
    """
    result = Elapsed()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
