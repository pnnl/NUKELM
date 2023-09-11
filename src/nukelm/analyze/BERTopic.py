# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
import logging
from copy import deepcopy
from typing import List, Tuple, Union

import hdbscan
import numpy as np
import pandas as pd
import umap
from bertopic import BERTopic as BERTopicParent
from bertopic._bertopic import check_is_fitted
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


logger = logging.getLogger(__name__)


class BERTopic(BERTopicParent):
    """Extension of `BERTopic` to expose additional hyperparameters."""

    def __init__(
        self,
        min_dist=0.0,
        umap_metric="euclidean",
        random_state=42,
        n_neighbors=15,
        n_components=5,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        hdbscan_metric="euclidean",
        alpha=1.0,
        cluster_selection_method="eom",
        vectorizer_kwargs=None,
        *args,
        **kwargs,
    ):
        """Initialize model."""
        super().__init__(*args, **kwargs)
        self.min_dist = min_dist
        self.umap_metric = umap_metric
        self.random_state = random_state

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.hdbscan_metric = hdbscan_metric
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method

        self.vectorizer_kwargs = vectorizer_kwargs if vectorizer_kwargs is not None else dict()

        if vectorizer_kwargs is not None:
            if "n_gram_range" in vectorizer_kwargs and vectorizer_kwargs["n_gram_range"] != self.n_gram_range:
                logger.warning("Overriding parameter 'n_gram_range' with value form 'vectorizer_kwargs'.")
                self.n_gram_range = vectorizer_kwargs["n_gram_range"]
            if "n_gram_range" in vectorizer_kwargs:
                vectorizer_kwargs["ngram_range"] = vectorizer_kwargs.pop("n_gram_range")
            self.vectorizer_model = CountVectorizer(**vectorizer_kwargs)

    def _reduce_dimensionality(
        self,
        embeddings: Union[np.ndarray, csr_matrix],
        y: Union[List[int], np.ndarray] = None,
    ) -> np.ndarray:
        """Reduce dimensionality of embeddings using UMAP and train a UMAP model.

        Args:
            embeddings: The extracted embeddings using the sentence transformer module.
            y: The target class for (semi)-supervised dimensionality reduction

        Returns:
            umap_embeddings: The reduced embeddings.
        """
        if isinstance(embeddings, csr_matrix):
            if self.umap_metric != "hellinger":
                logger.warning("CSR Matrix detected, overriding metric.")
            self.umap_model = umap.UMAP(
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                metric="hellinger",
                random_state=self.random_state,
                low_memory=self.low_memory,
            ).fit(embeddings, y=y)
        else:
            self.umap_model = umap.UMAP(
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                min_dist=self.min_dist,
                metric=self.umap_metric,
                random_state=self.random_state,
                low_memory=self.low_memory,
            ).fit(embeddings, y=y)
        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality with UMAP")
        return umap_embeddings

    def _cluster_embeddings(
        self, umap_embeddings: np.ndarray, documents: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Cluster UMAP embeddings with HDBSCAN.

        Args:
            umap_embeddings: The reduced sentence embeddings with UMAP.
            documents: Dataframe with documents and their corresponding IDs.

        Returns:
            documents: Updated dataframe with documents and their corresponding IDs
                       and newly added Topics.
            probabilities: The distribution of probabilities.
        """
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.hdbscan_metric,
            alpha=self.alpha,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        ).fit(umap_embeddings)
        documents["Topic"] = self.hdbscan_model.labels_

        if self.calculate_probabilities:
            probabilities = hdbscan.all_points_membership_vectors(self.hdbscan_model)
        else:
            probabilities = None

        self._update_topic_size(documents)
        logger.info("Clustered UMAP embeddings with HDBSCAN")
        return documents, probabilities

    def update_topics(
        self,
        docs: List[str],
        topics: List[int],
        vectorizer_kwargs=None,
    ):
        """Recalculate c-TF-IDF with new parameters.

        When you have trained a model and viewed the topics and the words that represent them,
        you might not be satisfied with the representation. Perhaps you forgot to remove
        stop_words or you want to try out a different n_gram_range. This function allows you
        to update the topic representation after they have been formed.
        Usage:
        In order to update the topic representation, you will need to first fit the topic
        model and extract topics from them. Based on these, you can update the representation:
        ```python
        model.update_topics(docs, topics, n_gram_range=(2, 3))
        ```
        YOu can also use a custom vectorizer to update the representation:
        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        model.update_topics(docs, topics, vectorizer_model=vectorizer_model)
        ```

        Args:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            vectorizer_kwargs: Pass in your own keyword arguments to CountVectorizer from scikit-learn
        """
        check_is_fitted(self)

        if vectorizer_kwargs is None:
            logger.warning("Reusing default vectorizer kwargs with no update.")
            vectorizer_kwargs = self.vectorizer_kwargs
        else:
            _vectorizer_kwargs = deepcopy(self.vectorizer_kwargs)
            _vectorizer_kwargs.update(vectorizer_kwargs)
            vectorizer_kwargs = _vectorizer_kwargs

        if "n_gram_range" in vectorizer_kwargs:
            vectorizer_kwargs["ngram_range"] = vectorizer_kwargs.pop("n_gram_range")
        self.vectorizer_model = CountVectorizer(**vectorizer_kwargs)

        documents = pd.DataFrame({"Document": docs, "Topic": topics})
        self._extract_topics(documents)
