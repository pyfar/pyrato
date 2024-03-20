#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'pyfar>=0.5.0',
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
    author="The pyfar developers",
    author_email='info@pyfar.org',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
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
    url='https://github.com/pyfar/pyrato',
    version='0.4.0',
    zip_safe=False,
    python_requires='>=3.8'
)
