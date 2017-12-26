import pickle
import gzip

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

def save_kaggle_sub(df, filename='sub.csv.gz'):
    """Saves the past dataframe with index=False, and enables GZIP compression if a '.gz' extension is passed.

    Keyword arguments:
    df -- The pandas DataFrame of the submission
    filename -- The filename to save the submission to. Autodetects '.gz'
    """
    if filename.endswidth('.gz'):
        compression = 'gzip'
    else:
        compression = None

    df.to_csv(filename, index=False, compression=compression)
