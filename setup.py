#!/usr/bin/env python3
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup(**generate_distutils_setup(
    packages=['fastlio_traversability_mapper'],
    package_dir={'': 'src'}
))
