# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
"""Utilities to provide a pseudo-exhaustive search to hyperopt.

From https://github.com/hyperopt/hyperopt/issues/200#issuecomment-507287308
"""
import numpy as np
from hyperopt import pyll
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll.base import as_apply, dfs
from hyperopt.pyll.stochastic import implicit_stochastic_symbols


class ExhaustiveSearchError(Exception):
    """Raise when exhaustive search is not possible."""

    pass


def validate_space_exhaustive_search(space):
    """Check whether a hyperopt search space is valid under exhaustive search.

    Args:
        space (hyperopt search space): Search space to validate

    Raises:
        ExhaustiveSearchError: Invalid stochastic symbol passed for exhaustive search.
    """
    supported_stochastic_symbols = ["randint", "quniform", "qloguniform", "qnormal", "qlognormal", "categorical"]
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError(
                    "Exhaustive search is only possible with the following stochastic symbols: "
                    + ", ".join(supported_stochastic_symbols)
                )


def suggest(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    """Perform a psuedo-exhaustive search by repeatedly sampling from the search space.

    Signature is similar to `hyperopt.rand.suggest` or `hyperopt.tpe.suggest`.

    Args:
        new_ids: See, e.g., `hyperopt.rand.suggest` or `hyperopt.tpe.suggest`.
        domain: See, e.g., `hyperopt.rand.suggest` or `hyperopt.tpe.suggest`.
        trials: See, e.g., `hyperopt.rand.suggest` or `hyperopt.tpe.suggest`.
        seed: See, e.g., `hyperopt.rand.suggest` or `hyperopt.tpe.suggest`.
        nbMaxSucessiveFailures (int, optional): Number of failures to allow before ending search. Defaults to 1000.
    """
    # Build a hash set for previous trials
    hashset = {
        hash(
            frozenset(
                (key, value[0]) if len(value) > 0 else ((key, None)) for key, value in trial["misc"]["vals"].items()
            )
        )
        for trial in trials.trials
    }

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={
                    domain.s_new_ids: [new_id],
                    domain.s_rng: rng,
                },
            )
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset((key, value[0]) if len(value) > 0 else ((key, None)) for key, value in vals.items()))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval
