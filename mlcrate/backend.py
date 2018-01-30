from warnings import warn

# Function wrapper that warns is deprecated, and call the function anyway
def _deprecated(func, old_name, new_name):
    def new_func(*args, **kwargs):
        message = '{}() has been deprecated in favour of {}() and will be removed soon'.format(old_name, new_name)
        warn(message)
        return func(*args, **kwargs)
    return new_func