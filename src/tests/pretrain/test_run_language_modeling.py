# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
import json
from pathlib import Path

from nukelm.pretrain.run_language_modeling import main as trainer


def test_pretrain(tmp_path):
    config_path = Path("src/tests/fixtures/test-config.json")
    with open(config_path) as fh:
        config = json.load(fh)

    config["output_dir"] = config["output_dir"].format(tmp_dir=tmp_path)
    tmp_config_path = Path(tmp_path) / "config.json"
    with open(tmp_config_path, "w") as fh:
        json.dump(config, fh)

    trainer(config_path=str(tmp_config_path))

    results_path = Path(tmp_path) / "all_results.json"
    assert results_path.exists()
    assert "perplexity" in json.loads(results_path.read_text())
