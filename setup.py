# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
from pathlib import Path

from setuptools import find_packages, setup


LICENSE = Path("Disclaimer.txt").read_text()
VERSION = "1.1.0"


entry_points = [
    "nukelm-pretrain = nukelm:run_language_modeling",
    "nukelm-finetune = nukelm:run_fine_tune",
    "nukelm-serve = nukelm:run_serve",
]


def requirements(infile):
    """Parse pip-formatted requirements file."""
    with open(infile) as f:
        packages = f.read().splitlines()
        return [pkg for pkg in packages if not pkg.startswith("#") and len(pkg.strip()) > 0]


setup(
    name="nukelm",
    version=VERSION,
    package_dir={"": "src"},
    packages=find_packages("src", exclude=["tests"]),
    description="Utilities for training and testing the NukeLM models.",
    author="DUDE/PNNL",
    license=LICENSE,
    entry_points={"console_scripts": entry_points},
    install_requires=requirements("requirements.txt"),
    extras_require={"dev": requirements("requirements-dev.txt")},
)
