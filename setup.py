#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "real-estate-predictor"
DESCRIPTION = "Regression Model package for real estate predictions."
URL = "https://github.com/julian-fong/real-estate-predictor-2.0"
EMAIL = "julian.fong@landpower.ca"
AUTHOR = "julian-fong"
REQUIRES_PYTHON = ">=3.8.0"

long_description = DESCRIPTION

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR
PACKAGE_DIR = ROOT_DIR / "real-estate-predictor"


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname, encoding="UTF-8") as fd:
        return fd.read().splitlines()


setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
