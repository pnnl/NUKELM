# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
import argparse
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List


LOG = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[2]


def call(cmd: List[str], **kwargs):
    """Run a subprocess command and raise if it fails.

    Args:
        cmd: List of command parts.
        **kwargs: Optional keyword arguments passed to `subprocess.run`.

    Raises:
        RuntimeError: If `subprocess.run` returns non-zero code.
    """
    LOG.debug(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    code = subprocess.run(cmd, **kwargs).returncode
    if code:
        raise RuntimeError(f"Error running command {cmd}. Return code was {code}")


def build_docs(output_path=None):
    """Build the project documentation."""
    if output_path is None:
        output_path = PROJECT_DIR / "docs"
    output_path = Path(output_path)
    shutil.rmtree(output_path / "source" / "API", ignore_errors=True)
    shutil.rmtree(output_path / "build", ignore_errors=True)
    call(
        [
            "sphinx-apidoc",
            "--module-first",
            "-o",
            str(output_path / "source" / "API"),
            str(PROJECT_DIR / "src" / "nukelm"),
        ]
    )
    call(
        [
            "sphinx-build",
            "-M",
            "html",
            str(PROJECT_DIR / "docs" / "source"),
            str(output_path / "_build"),
            "-a",
        ]
    )
    assert output_path / "_build"


def dir_path(path_to_check: str) -> Path:
    """Check if a command-line argument is a valid directory.

    https://stackoverflow.com/a/54547257

    Args:
        path (str): Command-line input to check.

    Raises:
        argparse.ArgumentTypeError: `path` is not a valid directory.

    Returns:
        pathlib.Path: A valid path to a directory.
    """
    path = Path(path_to_check)
    if path.is_dir() and path.exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
