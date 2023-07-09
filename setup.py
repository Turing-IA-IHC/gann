import os
import platform
import sys

import pkg_resources
from setuptools import find_packages, setup

def read_version(fname="gann/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]

requirements = []
setup(
    name="G0-gann",
    py_modules=["gann"],
    version=read_version(),
    description="Environment for Genetic Algorithms and Neural Networks",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.10",
    author="Gabriel Rojas (Gavit0) - G0",
    url="https://github.com/Turing-IA-IHC/gann",
    license="COPYRIGHT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)