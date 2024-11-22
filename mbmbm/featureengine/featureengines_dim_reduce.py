import numpy as np
from scipy.spatial.distance import pdist, squareform
from skbio.diversity import beta_diversity
from sklearn.decomposition import NMF, PCA, LatentDirichletAllocation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS

from mbmbm import logger
from mbmbm.featureengine.featureenginebase import FeatureReducerBase


class FeatureDimReducerNMF(FeatureReducerBase):
    """
    FeatureDimReducer based on Non-negative Matrix Factorization (NMF).

    :param n_components: The number of components to reduce to.
    :type n_components: int
    :param random_state: Random state for reproducibility.
    :type random_state: int
    """

    def __init__(self, n_components: int = 100, random_state: int = 42):
        super().__init__()

        self._n_components = n_components
        self._random_state = random_state
        logger.info(f"Initialized NMF with n_components={n_components} and random_state={random_state}")

    def _set_method(self):
        logger.debug("Setting NMF method.")
        return NMF(n_components=self._n_components, random_state=self._random_state)

    def get_reduction_info(self):
        """
        Get information about the reduction configuration.

        :return: A dictionary containing the number of components.
        :rtype: dict
        """
        return {"n_components": self._n_components}


class FeatureDimReducerPCA(FeatureReducerBase):
    """
    FeatureDimReducer based on Principal Component Analysis (PCA).

    :param n_components: The number of components to reduce to.
    :type n_components: int
    :param distance_metric: The distance metric to use for PCA, either 'euclidean' or 'jaccard'.
    :type distance_metric: str
    """

    def __init__(self, n_components: int = 100, distance_metric: str = "euclidean"):
        super().__init__()
        assert distance_metric in ["euclidean", "jaccard"], "Invalid distance metric"

        self._n_components = n_components
        self._distance_metric = distance_metric
        logger.info(f"Initialized PCA with n_components={n_components} and distance_metric={distance_metric}")

    def _set_method(self):
        logger.debug("Setting PCA method.")
        return PCA(n_components=self._n_components)

    def _fit(self, arr_input, arr_target):
        logger.debug("Calculating distance matrix for PCA fitting.")
        distance_matrix = pdist(arr_input, metric=self._distance_metric)
        distance_matrix = squareform(distance_matrix)
        logger.info("Fitting PCA model.")
        self._method.fit(distance_matrix)

    def _transform(self, arr_input):
        logger.debug("Calculating distance matrix for PCA transformation.")
        distance_matrix = pdist(arr_input, metric=self._distance_metric)
        distance_matrix = squareform(distance_matrix)
        logger.info("Transforming input using PCA model.")
        return self._method.transform(distance_matrix)

    def get_reduction_info(self):
        """
        Get information about the reduction configuration.

        :return: A dictionary containing the number of components.
        :rtype: dict
        """
        return {"n_components": self._n_components}


class FeatureDimReducerLDA(FeatureReducerBase):
    """
    FeatureDimReducer based on Linear Discriminant Analysis (LDA).

    :param n_components: The number of components to reduce to.
    :type n_components: int
    """

    def __init__(self, n_components: int = 100):
        super().__init__()

        self._n_components = n_components
        logger.info(f"Initialized LDA with n_components={n_components}")

    def _set_method(self):
        logger.debug("Setting LDA method.")
        return LinearDiscriminantAnalysis(n_components=self._n_components)

    def get_reduction_info(self):
        """
        Get information about the reduction configuration.

        :return: A dictionary containing the number of components.
        :rtype: dict
        """
        return {"n_components": self._n_components}


class FeatureDimReducerNMDS(FeatureReducerBase):
    """
    FeatureDimReducer based on Non-metric Multidimensional Scaling (NMDS).

    :param n_components: The number of components to reduce to.
    :type n_components: int
    :param random_state: Random state for reproducibility.
    :type random_state: int
    """

    def __init__(self, n_components: int = 100, random_state: int = 42):
        super().__init__()

        self._n_components = n_components
        self._random_state = random_state
        logger.info(f"Initialized NMDS with n_components={n_components} and random_state={random_state}")

    def _set_method(self):
        logger.debug("Setting NMDS method.")
        return MDS(n_components=self._n_components, dissimilarity="precomputed", random_state=self._random_state)

    def _fit(self, arr_input, arr_target):
        logger.debug("Calculating distance matrix for NMDS fitting.")
        distance_matrix = beta_diversity("braycurtis", arr_input)
        distance_matrix = np.asarray(distance_matrix.data)
        logger.info("Fitting NMDS model.")
        self._method.fit_transform(distance_matrix)

    def _transform(self, arr_input):
        logger.debug("Calculating distance matrix for NMDS transformation.")
        distance_matrix = beta_diversity("braycurtis", arr_input)
        distance_matrix = np.asarray(distance_matrix.data)
        logger.info("Transforming input using NMDS model.")
        return self._method.fit_transform(distance_matrix)

    def get_reduction_info(self):
        """
        Get information about the reduction configuration.

        :return: A dictionary containing the number of components.
        :rtype: dict
        """
        return {"n_components": self._n_components}


class FeatureDimReducerLatentDirichletAllocation(FeatureReducerBase):
    """
    FeatureDimReducer based on Latent Dirichlet Allocation (LDA).

    :param n_components: The number of components to reduce to.
    :type n_components: int
    :param max_iter: The maximum number of iterations for LDA.
    :type max_iter: int
    :param learning_method: The learning method for LDA, either 'batch' or 'online'.
    :type learning_method: str
    """

    def __init__(self, n_components: int = 100, max_iter: int = 10, learning_method: str = "batch"):
        super().__init__()

        self._n_components = n_components
        self._max_iter = max_iter
        self._learning_method = learning_method
        logger.info(
            f"Initialized LatentDirichletAllocation with n_components={n_components}, max_iter={max_iter}, "
            f"learning_method={learning_method}"
        )

    def _set_method(self):
        logger.debug("Setting Latent Dirichlet Allocation method.")
        return LatentDirichletAllocation(
            n_components=self._n_components, max_iter=self._max_iter, learning_method=self._learning_method
        )

    def _fit(self, arr_input, arr_target=None):
        logger.info("Fitting Latent Dirichlet Allocation model.")
        self._method.fit(arr_input)

    def _transform(self, arr_input):
        logger.info("Transforming input using Latent Dirichlet Allocation model.")
        return self._method.transform(arr_input)

    def get_reduction_info(self):
        """
        Get information about the reduction configuration.

        :return: A dictionary containing the number of components, max iterations, and learning method.
        :rtype: dict
        """
        return {
            "n_components": self._n_components,
            "max_iter": self._max_iter,
            "learning_method": self._learning_method,
        }
