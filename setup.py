from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='mlcrate',
    version='0.2.0',
    description='A collection of handy python tools and functions, mainly for ML and Kaggle.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['mlcrate'],
    author='Mikel Bober-Irizar',
    author_email='mikel@mxbi.net',
    url='https://github.com/mxbi/mlcrate',
    license='MIT',
    install_requires=['numpy', 'pandas', 'pathos', 'tqdm'],
    classifiers=['License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Development Status :: 4 - Beta',
        ]
)
