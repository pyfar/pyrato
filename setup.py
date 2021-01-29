#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Marco Berzborn - Institute of Technical Acoustics",
    author_email='marco.berzborn@akustik.rwth-aachen.de',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Scientists',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Collection of functions for commonly used calculcations in room acoustics",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='roomacoustics',
    name='roomacoustics',
    packages=find_packages(include=['roomacoustics']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://git.rwth-aachen.de/mbe/roomacoustics',
    version='0.2.1',
    zip_safe=False,
)
