import signal
from contextlib import contextmanager
from z3 import unknown

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Solver timed out")

    original_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def solve_with_timeout(solver, timeout_ms):
    """Wrapper for Z3 solver with both soft and hard timeouts"""
    timeout_seconds = timeout_ms / 1000
    
    solver.set("timeout", timeout_ms)
    
    try:
        with timeout_handler(timeout_seconds):
            result = solver.check()
            return result
    except TimeoutException:
        solver.reset()
        return unknown
