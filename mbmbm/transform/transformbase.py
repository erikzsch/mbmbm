from abc import ABC

import numpy as np

from mbmbm import logger


class TransformBase(ABC):
    """
    Base class for all Dataset transforms.

    A transform is a class that implements the method :meth:`transform`, which takes a NumPy array as input,
    modifies it according to a specific aspect, and returns the transformed array. Additionally, this method
    returns information about rows and columns removed during the transformation process.

    The transform process should be explicitly defined in any subclass, and the aspect of the transformation
    should be clearly indicated in the class name.

    Methods
    -------
    fit(arr: np.ndarray)
        Prepares the transformation using the provided array.

    transform(arr: np.ndarray) -> (np.ndarray, list or None, list or None)
        Applies the transformation to the input array and returns the transformed array,
        along with the indices of removed rows and columns if applicable.
    """

    def __init__(self):
        """
        Initialize the base transform class.

        This constructor does not take any parameters, but can be overridden by subclasses if necessary.
        """
        logger.debug("Initializing TransformBase class.")
        ...

    def fit(self, arr: np.ndarray):
        """
        Fit the transformation model using the provided NumPy array.

        This method should be used to prepare the transformation, such as calculating necessary statistics
        or storing relevant information. This base method should be implemented by subclasses.

        Parameters
        ----------
        arr : np.ndarray
            The input NumPy array used for fitting the transformation.
        """
        logger.info("Fitting the transformation model.")
        ...

    def transform(self, arr: np.ndarray) -> (np.ndarray, list or None, list or None):
        """
        Apply the transformation to the input NumPy array.

        The transformation will alter the input array according to the rules defined by the subclass.
        The transformed array, along with the indices of removed rows and columns (if applicable), will be returned.
        This method must be overridden by subclasses.

        Parameters
        ----------
        arr : np.ndarray
            The input NumPy array to be transformed.

        Returns
        -------
        np.ndarray
            The transformed NumPy array.
        list or None
            A list of indices representing removed rows, or None if no rows are removed.
        list or None
            A list of indices representing removed columns, or None if no columns are removed.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        logger.error("Transform method not implemented in base class.")
        raise NotImplementedError("The transform method must be implemented by subclasses.")
