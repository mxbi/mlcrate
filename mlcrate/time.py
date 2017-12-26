import time
import datetime

class TimeTracker:
    def __init__(self):
        self.times = {}
        self.add(0)

    def __getitem__(self, key):
        return self.times[key]

    def add(self, key):
        self.times[key] = time.time()

    def elapsed(self, key):
        return time.time() - self.times[key]

def get_current_str_time():
    """Returns the current time as a string in the format 'YYYY_MM_DD_HH_MM_SS'. Useful for timestamping filenames etc."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")
