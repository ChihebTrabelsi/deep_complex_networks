#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md') as f:
    DESCRIPTION = f.read()


setup(
    name='DeepComplexNetworks',
    version='1',
    license='MIT',
    long_description=DESCRIPTION,
    packages=find_packages() + find_packages('musicnet/'),
    package_dir={'musicnet': 'musicnet/musicnet'},
    scripts=['scripts/run.py', 'scripts/training.py', 'musicnet/scripts/train.py',
             'musicnet/scripts/resample.py'],
    install_requires=[
        "numpy", "scipy", "sklearn", "Theano", "keras", "intervaltree",
        "resampy", "mimir", "kerosene"]
)
