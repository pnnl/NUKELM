# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
from datasets import load_from_disk

from nukelm.analyze.serve import main as server


def test_serve_documents(tmp_path):
    server(
        "distilbert-base-uncased-finetuned-sst-2-english",
        "src/tests/fixtures/example-osti-abstracts.csv",
        tmp_path,
        use_cuda=False,
    )
    dataset = load_from_disk(tmp_path)
    for strategy in ["CLS", "MEAN", "MAX"]:
        assert strategy in dataset.features
    assert "predicted-scores" in dataset.features
    assert "predicted-label" in dataset.features
