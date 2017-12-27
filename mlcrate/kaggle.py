def save_sub(df, filename='sub.csv.gz'):
    """Saves the passed dataframe with index=False, and enables GZIP compression if a '.gz' extension is passed.

    Keyword arguments:
    df -- The pandas DataFrame of the submission
    filename -- The filename to save the submission to. Autodetects '.gz'
    """
    if filename.endswidth('.gz'):
        compression = 'gzip'
    else:
        compression = None

    df.to_csv(filename, index=False, compression=compression)
