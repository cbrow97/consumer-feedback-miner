import pathlib
import re

from setuptools import setup, find_packages
from typing import Dict, List

# This code computes the version string for the package
VERSIONFILE = "src/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


def add_pytyped_files(
    base_dir: str, pkg_data: Dict[str, List[str]] = {}
) -> Dict[str, List[str]]:
    """Scans packages for py.typed files and adds it to package_data.
    This allows for users of this library to use mypy without warnings.
    """
    packages = find_packages(base_dir)
    for package in packages:
        # find py.typed files
        # package is foo.bar, replace = foo/bar
        pytyped_file = pathlib.Path(base_dir, package.replace(".", "/"), "py.typed")
        if not pytyped_file.exists():
            continue

        # this means pytyped_file.exists()
        data = pkg_data.get(package)
        if data:
            data.append("py.typed")
        else:
            pkg_data[package] = ["py.typed"]

    return pkg_data


package_data = add_pytyped_files("src")

setup(
    name="feedback_analyzer",
    version=version_string,
    author="Colton Brown",
    description="A package for mining and analyzing consumer feedback from third party platforms",
    url="https://https://github.com/cbrow97/consumer-feedback-analyzer",
    packages=find_packages("src"),
    package_data=package_data,
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy"
    ],
    extras_require={
        "dev": [
        ]
    },
    python_requires=">=3.7.*,<3.9.*",
)
