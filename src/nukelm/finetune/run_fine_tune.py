# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
"""Wrap transformers text classification example script to run from existing Python process."""
import logging
import sys
from pathlib import Path
from typing import Union

from nukelm.finetune.run_glue import main as run_glue


LOG = logging.getLogger(__name__)


def main(config_path: Union[Path, str] = None) -> dict:
    """Wrap transformers text classification example script.

    Args:
        config_path (Union[Path, str], optional): Path to a configuration file. Defaults to None.

    Raises:
        ValueError: transformers version does not match included "run_glue.py" script version.

    Returns:
        dict: evaluation results, as `from run_glue.main`.
    """
    saved_args = sys.argv
    if config_path is None:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            config_path = sys.argv[1]

    config_path = str(config_path)

    sys.argv = [str(sys.argv[0]), config_path]

    try:
        return run_glue()
    finally:
        sys.argv = saved_args


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(module)s.%(funcName)s.L%(lineno)d - %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()
