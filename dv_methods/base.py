"""Definition of base class for data valuation methods."""
from abc import ABC, abstractmethod
import numpy as np


class DVMethodBase(ABC):
    """Base class for methods to compute data values."""

    NAME: str
    URL: str

    @abstractmethod
    def __init__(self, X_base=None, y_base=None):  # noqa
        """Init the data-valuation method.

        All necessary values for the call of 'predict_dv()' should be set here via 'streamlit.sidebar'.
        E.g:

            model = ui.sidebar_components.model_selector()
            size_hidden_layer = ui.sidebar_components.size_hidden_layer_selector()

        Args:
            X_base (np.ndarray): The baseline features of shape (n_samples, n_features, ).
            y_base: (np.ndarray) The baseline labels of shape (n_samples, ).
        """
        self.X_base = X_base
        self.y_base = y_base

    @abstractmethod
    def predict_dv(self, X, y) -> (np.ndarray, callable):  # noqa
        """Returns array of data values and a base-line model trained on the base data set.

        The function should compute the value for each instance 'i' in 'X' w.r.t. 'self.X_base' and 'self.y_base'.

        Args:
            X (np.ndarray): The features of shape (n_samples, n_features, ).
            y (np.ndarray): The labels of shape (n_samples, ).

        Returns:
            data_values (np.ndarray): Date values of each instance in 'X' of shape (n_samples, )
            base_model (Callable): The baseline model trained on ('self.X_base', 'self.y_base')
        """
