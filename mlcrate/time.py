import time
import datetime

class Timer:
    """A class for tracking timestamps and time elapsed since events. Useful for profiling code.

    Usage:
    >>> t = TimeTracker()
    >>> t.elapsed(0) # Seconds since the timetracker was initialised
    >>> t.add('func') # Save the current timestamp as 'func'
    >>> t.elapsed('func') # Seconds since 'func' was added
    >>> t['func'] # Get the absolute timestamp of 'func' for other uses
    """
    def __init__(self):
        self.times = {}
        self.add(0)

    def __getitem__(self, key):
        return self.times[key]

    def add(self, key):
        """Add the current time to the index with the specified key"""
        self.times[key] = time.time()

    def elapsed(self, key):
        """Get the time passed in seconds since the specified key was added to the index"""
        return time.time() - self.times[key]

def str_time_now():
    """Returns the current time as a string in the format 'YYYY_MM_DD_HH_MM_SS'. Useful for timestamping filenames etc."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")
