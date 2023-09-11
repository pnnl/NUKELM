# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
import hyperopt

from nukelm.hyperopt_utils import ExhaustiveSearchError, suggest, validate_space_exhaustive_search


VALID_SPACE = {
    "Batch Size": hyperopt.hp.choice("batch_size", [16, 64]),
    "Learning Rate": hyperopt.hp.choice("learning_rate", [1e-5, 2e-5, 5e-5]),
}
INVALID_SPACE = {
    "Batch Size": hyperopt.hp.choice("batch_size", [16, 64]),
    "Learning Rate": hyperopt.hp.loguniform("learning_rate", 1e-5, 5e-5),
}


def test_validate_exhaustive_search():
    try:
        validate_space_exhaustive_search(VALID_SPACE)
    except ExhaustiveSearchError:
        assert False, "Valid space marked invalid."
    try:
        validate_space_exhaustive_search(INVALID_SPACE)
    except ExhaustiveSearchError:
        pass
    else:
        assert False, "Invalid space marked valid."


def test_exhaustive_suggest():
    _ALL_ARGS = []

    def dummy_objective(args):
        """Make list of all args from exhaustive search."""
        _ALL_ARGS.append(args)
        return 0

    _ = hyperopt.fmin(
        fn=dummy_objective,
        space=VALID_SPACE,
        trials=hyperopt.Trials(),
        algo=suggest,
        max_evals=1000,
        show_progressbar=False,
    )
    NUM_ARGS = len(_ALL_ARGS)

    assert NUM_ARGS == 6
