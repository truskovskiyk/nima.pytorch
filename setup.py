# -*- coding: utf-8 -*-

import re

from setuptools import find_packages, setup


with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


with open("nima/__init__.py") as f:
    txt = f.read()
    try:
        version = re.findall(r'^__version__ = "([^"]+)"\r?$', txt, re.M)[0]
    except IndexError:
        raise RuntimeError("Unable to determine version.")

setup(
    name="nima",
    version=version,
    python_requires=">=3.7.0",
    install_requires=install_requires,
    include_package_data=True,
    description="Neural IMage Assessment",
    long_description=readme,
    long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
    packages=find_packages(),
    entry_points={"console_scripts": "nima-cli=nima.cli:main"},
)
