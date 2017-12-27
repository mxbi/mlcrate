# mlcrate
A collection of handy python tools and functions, mainly for ML and Kaggle.

The methods in this package aren't revolutionary, and most of them are very simple. They are largely bunch of 'macro' functions which I often end up rewriting across multiple projects, all in one place and easily accessible to make my life easier and let me write code faster in the future. Hopefully, they can be some use to others in the community too.

This package has been tested with Python 3.5+, but should work with all versions of Python 3. Python 2 is not officially supported (although most of the functions would work in theory.)

## Installation

Clone the repo and run `python setup.py install` within the top-level folder to install mlcrate.

## Features

Here is an overview of the functions in the package. Click the links to view detailed docs (docstrings)

- Simple pickle wrapper for fast save/load of arbitrary python objects (with optional compression).
Works with numpy, pandas, etc. and objects >4GB. Also cross-compatible with standard pickle dump/load.

[mlcrate.save()](https://github.com/mxbi/mlcrate/blob/df66daf0a9e7078058aa65a7f42f9509f0d2d300/mlcrate/__init__.py#L9), [mlcrate.load()](https://github.com/mxbi/mlcrate/blob/df66daf0a9e7078058aa65a7f42f9509f0d2d300/mlcrate/__init__.py#L24)  

```python
>>> import mlcrate as mlc
>>> x = [1, 2, 3, 4]

>>> mlc.save(x, 'x.pkl.gz')  # Saves using GZIP when .gz extension is used

>>> mlc.load('x.pkl.gz')
[1, 2, 3, 4]
```

More has been currently added to the package but is not documented here in the README. Please look at the comprehensive docstrings in the source for more info for now.
