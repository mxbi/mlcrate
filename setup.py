from setuptools import setup

setup(
    name='mlcrate',
    version='0.1.0',
    description='A collection of handy python tools and functions, mainly for ML and Kaggle.',
    long_description="For more info, see https://github.com/mxbi/mlcrate",
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
