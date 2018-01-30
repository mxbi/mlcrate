from . import time

def save_sub(df, filename='sub_{}.csv.gz'):
    """Saves the passed dataframe with index=False, and enables GZIP compression if a '.gz' extension is passed.
    If '{}' exists in the filename, this is replaced with the current time from mlcrate.time.now()

    Keyword arguments:
    df -- The pandas DataFrame of the submission
    filename -- The filename to save the submission to. Autodetects '.gz'
    """
    if '{}' in filename:
        filename = filename.format(time.now())
    if filename.endswith('.gz'):
        compression = 'gzip'
    else:
        compression = None

    df.to_csv(filename, index=False, compression=compression)
