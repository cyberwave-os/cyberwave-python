import threading
import time


class TimeReference:
    """
    Class to store and update time values.
    """
    def __init__(self) -> None:
        """
        Initialize the time reference.
        """
        self.lock = threading.Lock()
        self.time = time.time()
        self.time_monotonic = time.monotonic()

    def update(self) -> tuple[float, float]:
        """
        Update the time values.
        """
        with self.lock:
            self.time = time.time()
            self.time_monotonic = time.monotonic()
            return self.time, self.time_monotonic

    def read(self) -> tuple[float, float]:
        """
        Read the time values.
        """
        with self.lock:
            return self.time, self.time_monotonic
