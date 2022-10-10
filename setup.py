#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>=1.14.0',
    'scipy>=1.5.0',
    'matplotlib']

setup_requirements = ['pytest-runner', ]

test_requirements = [
    'pytest',
    'bump2version',
    'wheel',
    'watchdog',
    'flake8',
    'tox',
    'coverage',
    'Sphinx',
    'twine'
]

setup(
    author="Marco Berzborn - Institute for Hearing Technology and Acoustics",
    author_email='marco.berzborn@akustik.rwth-aachen.de',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Collection of functions commonly used in room acoustics",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='pyrato',
    name='pyrato',
    packages=find_packages(include=['pyrato']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mberz/pyrato',
    version='0.3.2',
    zip_safe=False,
)
