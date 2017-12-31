import pickle
import gzip
import os

from . import time, kaggle, xgb#, sklearn

__version__ = '0.0.0a1'

def save(data, filename):
    """Pickles the passed data (with the highest available protocol) to disk using the passed filename.
    If the filename ends in '.gz' then the data will additionally be GZIPed before saving.

    Keyword arguments:
    data -- The python object to pickle to disk (use a dict or list to save multiple objects)
    filename -- String with the relative filename to save the data to. By convention should end in '.pkl' or 'pkl.gz'
    """
    if filename.endswith('.gz'):
        fp = gzip.open(filename, 'wb')
    else:
        fp = open(filename, 'wb')

    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    """Loads data saved with save() (or just normally saved with pickle). Uses gzip if filename ends in '.gz'

    Keyword arguments:
    filename -- String with the relative filename of the pickle at load.
    """
    if filename.endswith('.gz'):
        fp = gzip.open(filename, 'rb')
    else:
        fp = open(filename, 'rb')

    return pickle.load(fp)

class LinewiseCSVWriter:
    def __init__(self, filename, header=None, sync=True, append=False):
        """CSV Writer which writes a single line at a time, and by default syncs to disk after every line.
        This is useful for eg. log files, where you want progress to appear in the file as it happens (instead of being written to disk when python exists)
        Data should be passed to the writer as an iterable, as conversion to string and so on is done within the class.

        Keyword arguments:
        filename -- the csv file to write to
        header (default: None) -- An iterator (eg. list) containing an optional CSV header, which is written as the first line of the file.
        sync (default: True) -- Flush and sync the output to disk after every write operation. This means data appears in the file instantly instead of being buffered
        append (default: False) -- Append to an existing CSV file. By default, the csv file is overwritten each time.
        """
        self.autoflush = sync
        self.len = None

        mode = 'a' if append else 'w'
        self.f = open(filename, mode)

        if header:
            self.write(header)

    def __del__(self):
        self.flush()
        self.f.close()

    def write(self, data):
        """Write a line of data to the csv file - data should be an iterable."""
        if self.len is None:
            self.len = len(data)
        assert len(data) == self.len, 'Length of passed data does not match previous rows'

        text = '"' + '","'.join([str(x).replace('"', '""') for x in data]) + '"\n'

        self.f.write(text)
        if self.autoflush:
            self.flush()

    def flush(self):
        """Manually flush the csv. Useless if flush=True in the class, as this is called after every write anyway."""
        self.f.flush()
        os.fsync(self.f.fileno())


class SuperPool:
    def __init__(self, n_cpu=-1):
        from multiprocessing import cpu_count
        from pathos.multiprocessing import ProcessPool
        import tqdm

        self.tqdm = tqdm

        if n_cpu == -1:
            n_cpu = cpu_count()

        self.n_cpu = n_cpu
        self.pool = ProcessPool(n_cpu)

    def __del__(self):
        self.pool.close()

    def map(self, func, array, chunksize=16, description=''):
        res = []

        def func_tracked(args):
            x, i = args
            return func(x), i

        array_tracked = zip(array, range(len(array)))

        desc = '[mlcrate] {} CPUs{}'.format(self.n_cpu, ' - {}'.format(description) if description else '')
        for out in self.tqdm.tqdm(self.pool.uimap(func_tracked, array_tracked, chunksize=chunksize), total=len(array), desc=desc, smoothing=0.05):
            res.append(out)

        # Sort based on i but return only the actual function result
        actual_res = [r[0] for r in sorted(res, key=lambda r: r[1])]

        return actual_res

    def exit(self):
        self.pool.close()
        self.pool.join()
