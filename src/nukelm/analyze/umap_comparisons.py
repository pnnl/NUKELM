# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import datasets
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import rc

from nukelm.utils import dir_path


LOG = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).parents[3]
PLOT_KWARGS = {
    "linestyle": "None",
    "marker": ".",
    "alpha": 0.5,
}  # type: dict
UMAP_KWARGS = {
    "n_neighbors": 15,
    "n_components": 2,
    "min_dist": 0.1,
    "metric": "euclidean",
    "random_state": 42,
}  # type: dict

LABEL_MAP = {
    "nuke": "NFC-Related",
    "not-nuke": "Other",
}


def plot_points(
    points: Sequence[Union[np.ndarray]],
    labels: Optional[Sequence[str]] = None,
    titles: Optional[Sequence[str]] = None,
    label_map: Dict[str, str] = None,
    legend: bool = True,
    **plot_kwargs: dict,
) -> plt.Figure:
    """Make scatterplot(s).

    Args:
        points (Sequence[Union[np.ndarray]]): Sequences of (x, y) coordinates.
        labels (Sequence[str]): Labels for each collection in `points`. Defaults to None.
        titles (Optional[Sequence[str]], optional): Titles for the plot(s). Defaults to None.
        label_map (Dict[str, str]): Re-label instances with print-ready names. Defaults to `LABEL_MAP`.
        legend (bool): Whether to include a legend. Defaults to True.

    Returns:
        plt.Figure: Figure containing the plot(s).
    """
    # enable LaTeX formatting in titles
    rc("text", usetex=True)

    if label_map is None:
        label_map = LABEL_MAP

    n_plots = len(points)
    if labels is not None:
        assert len(points) == len(labels)
        for _points, _labels in zip(points, labels):
            assert len(_points) == len(_labels)
    if titles is not None:
        assert len(titles) == n_plots
    else:
        titles = [""] * n_plots

    fig, axes = plt.subplots(1, n_plots, figsize=(3.5 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]

    if labels is not None:
        _unique_labels = set()
        for _labels in labels:
            for label in _labels:
                _unique_labels.add(label)
        unique_labels = sorted(list(_unique_labels))

    _mins, _maxs = [], []

    for i in range(n_plots):
        ax, _points, title = axes[i], points[i], titles[i]
        if labels is not None:
            _labels = labels[i]
            idx = {}
            for class_name in unique_labels:
                idx[class_name] = [i for i, label in enumerate(_labels) if label == class_name]
            for class_name in unique_labels:
                ax.plot(
                    _points[idx[class_name], 0],
                    _points[idx[class_name], 1],
                    label=label_map[class_name],
                    **plot_kwargs,
                )
            if legend:
                ax.legend()
        else:
            ax.plot(_points[:, 0], _points[:, 1], **plot_kwargs)

        _ymin, _ymax = ax.get_ylim()
        _mins.append(_ymin)
        _maxs.append(_ymax)
        _xmin, _xmax = ax.get_xlim()
        _mins.append(_xmin)
        _maxs.append(_xmax)

        if title:
            ax.set_title(title)

    _min = min(_mins) * 1.1
    _max = max(_maxs) * 1.1
    for ax in axes:
        ax.axis("equal")
        ax.set_ylim((_min, _max))
        ax.set_xlim((_min, _max))

    fig.tight_layout()

    return fig


def main(
    dataset_paths: Sequence[Union[str, Path]],
    map_indices: Sequence[int],
    data_column: str,
    titles: Optional[Sequence[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    **umap_kwargs: dict,
) -> None:
    """Compare embeddings using UMAP dimensionality reduction.

    Args:
        dataset_paths (Sequence[Union[str, pathlib.Path]]): Path(s) to directories created with
            `datasets.dataset.save_to_disk`.
        map_indices (Sequence[int]): Index(es) of datasets to use for each UMAPs.
            Should be same length as dataset-paths.
        data_column (str): Which pooling strategy to use: one of ["CLS", "MEAN", "MAX"].
        titles (Optional[Sequence[str]], optional): [description]. Defaults to None.
        output_path (Optional[Union[str, pathlib.Path]], optional): Where to save the figure.
            If omitted, will display instead. Defaults to None.
    """
    assert len(dataset_paths) == len(map_indices)
    if titles is not None:
        assert len(titles) == len(dataset_paths)

    LOG.debug("Loading datasets")
    _datasets = [datasets.load_from_disk(path) for path in dataset_paths]
    labels = [dataset["label"] for dataset in _datasets]

    LOG.debug("Computing UMAPs")
    _umap_kwargs = UMAP_KWARGS.copy()
    _umap_kwargs.update(umap_kwargs)
    mappers_dict = {}
    for i in set(map_indices):
        mappers_dict[i] = umap.UMAP(**_umap_kwargs).fit(_datasets[i][data_column])
    mappers = [mappers_dict[i] for i in map_indices]

    LOG.debug("Computing projections.")
    _points = [mapper.transform(dataset[data_column]) for mapper, dataset in zip(mappers, _datasets)]

    LOG.debug("Making figure")
    fig = plot_points(
        _points,
        labels,
        titles,
        **PLOT_KWARGS,
    )

    if output_path is not None:
        LOG.debug("Saving figure")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(module)s.%(funcName)s.L%(lineno)d - %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Compare embeddings using UMAP dimensionality reduction.")
    parser.add_argument(
        "--dataset-paths",
        action="extend",
        nargs="+",
        type=dir_path,
        required=True,
        help="Path(s) to directories created with `datasets.dataset.save_to_disk`.",
    )
    parser.add_argument(
        "--map-indices",
        action="extend",
        nargs="+",
        type=int,
        default=None,
        help="Index(es) of datasets to use for each UMAPs. Should be same length as dataset-paths.",
    )
    parser.add_argument(
        "--titles",
        action="extend",
        nargs="*",
        type=str,
        default=None,
        help="Title(s) for the plots. Should be same length as dataset-paths (optional).",
    )
    parser.add_argument(
        "--data-column", choices=["CLS", "MEAN", "MAX"], default="CLS", help="Which pooling strategy to use."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to save the figure. If omitted, will display instead.",
    )
    args = parser.parse_args()

    if args.map_indices is None:
        args.map_indices = [0] * len(args.dataset_paths)
    if len(args.dataset_paths) != len(args.map_indices):
        parser.error("dataset-paths and map-indices must be the same length.")
    if args.titles is not None and len(args.titles) != len(args.dataset_paths):
        parser.error("dataset-paths and titles must be the same length.")

    main(args.dataset_paths, args.map_indices, args.data_column, args.titles, args.output_path)
