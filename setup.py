# -*- coding: utf-8 -*-
"""Setup script for pompy package."""

from setuptools import setup
from os import path
from io import open


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'docs', 'description.rst'), encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pompy',
    version='0.1.1',
    description='Puff-based odour plume model',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Matt Graham',
    license='MIT',
    url='https://github.com/InsectRobotics/pompy',
    packages=['pompy'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
    ],
    keywords='simulation plume navigation',
    project_urls={
        'Documentation': 'https://pompy-docs.readthedocs.io/en/latest/',
    },
    include_package_data=True
)
